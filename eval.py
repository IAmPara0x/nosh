"""LLM-as-judge evaluation for the nosh NL->bash model, backed by vLLM.

Runs in two sequential phases (only one model on the GPU at a time):

  Phase 1  predict   — generate bash commands with the nosh model.
  Phase 2  judge     — score commands with the judge model.

vLLM gives us batched inference with continuous batching + automatic prefix
caching across requests (the shared system+few-shot prefix is encoded once and
reused for every row in the batch). For a 50-row CSV this is dramatically
faster than the per-row transformers loop.

Run:
    python eval.py 009.csv
    python eval.py 009.csv --has-expected            # skip phase 1, judge expected directly
    python eval.py 009.csv --phase predict           # only generate, save predictions
    python eval.py 009.csv --phase judge             # only judge an existing predictions file
"""

import argparse
import gc
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM as VLLM, SamplingParams

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"

RUBRIC_COLUMNS = [
    "prompt", "command", "expected",
    "exact_match", "formatting", "syntax_valid",
    "tool", "flags", "args", "constraints", "safety",
    "overall", "reasoning",
]


# ----- vLLM wrapper -----

class VLLMRunner:
    """Wraps a vLLM model with a fixed system + few-shot prefix.

    vLLM's prefix caching is automatic: every chat() request shares the same
    leading tokens, so vLLM encodes the system+few-shot prefix once and reuses
    its KV cache for all requests in the batch.
    """

    def __init__(self, model_id, system_prompt_path, examples_path,
                 max_model_len=4096, gpu_memory_utilization=0.9,
                 quantization=None, dtype="auto"):
        with open(system_prompt_path) as f:
            self.system_prompt = f.read().strip()
        with open(examples_path) as f:
            self.fewshots = json.load(f)

        kwargs = dict(
            model=model_id,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype=dtype,
        )
        if quantization:
            kwargs["quantization"] = quantization
        self.llm = VLLM(**kwargs)

    def _conversation(self, user_text):
        return ([{"role": "system", "content": self.system_prompt}]
                + self.fewshots
                + [{"role": "user", "content": user_text}])

    def generate_batch(self, user_texts, temperature=0.0, max_tokens=128):
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0.0 else 0.95,
        )
        conversations = [self._conversation(t) for t in user_texts]
        # Qwen3 supports a `thinking` mode; disable it so the model emits the
        # bash command / judge JSON directly without a <think>...</think> block.
        outputs = self.llm.chat(
            conversations,
            sampling_params=sampling,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        return [o.outputs[0].text.strip() for o in outputs]

    def shutdown(self):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----- deterministic pre-checks -----

MARKDOWN_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"^\s*[`']?bash[`']?\s*$", re.MULTILINE),
]


def check_formatting(command: str) -> bool:
    """True iff the output looks like bare bash (no markdown / json / prose preamble)."""
    if not command:
        return False
    stripped = command.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return False
    for pat in MARKDOWN_PATTERNS:
        if pat.search(stripped):
            return False
    if stripped.lower().startswith(("here is", "here's", "sure,", "to ", "you can")):
        return False
    return True


def check_syntax(command: str) -> bool:
    """True iff `bash -n` accepts the command without parse errors."""
    if not command.strip():
        return False
    try:
        result = subprocess.run(
            ["bash", "-n"], input=command, text=True,
            capture_output=True, timeout=5,
            errors="replace",  # low-quality model output can contain non-UTF-8 bytes
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, UnicodeDecodeError):
        return False


def normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


# ----- judge output parsing -----

JUDGE_KEYS = ["tool", "flags", "args", "constraints", "safety", "overall", "reasoning"]


def parse_judge_output(raw: str):
    """Robust parse: direct json.loads, else extract the outermost {...} block."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if all(k in obj for k in JUDGE_KEYS):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if all(k in obj for k in JUDGE_KEYS):
                return obj
        except json.JSONDecodeError:
            return None
    return None


# ----- aggregation -----

def aggregate(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {}
    judge_ok = df["overall"].notna()
    judged = df[judge_ok]
    out = {
        "n_rows": n,
        "n_judged": int(judge_ok.sum()),
        "exact_match_rate": round(df["exact_match"].mean(), 3),
        "formatting_rate": round(df["formatting"].mean(), 3),
        "syntax_valid_rate": round(df["syntax_valid"].mean(), 3),
    }
    if len(judged):
        out.update({
            "tool_rate": round(judged["tool"].mean(), 3),
            "safety_rate": round(judged["safety"].mean(), 3),
            "mean_flags": round(judged["flags"].mean(), 3),
            "mean_args": round(judged["args"].mean(), 3),
            "mean_constraints": round(judged["constraints"].mean(), 3),
            "mean_overall": round(judged["overall"].mean(), 3),
            "pct_overall_ge_4": round((judged["overall"] >= 4).mean(), 3),
        })
    return out


# ----- main -----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("file_name", help="CSV filename inside ./datasets (e.g. 009.csv)")
    p.add_argument("--nosh-model", default="./checkpoints/qwen2.5-coder-1.5b-nosh-merged")
    p.add_argument("--judge-model", default="unsloth/Qwen3-14B-unsloth-bnb-4bit")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default ./results/<file_name>)")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--has-expected", action="store_true",
                   help="CSV's `command` column already holds the candidate command; skip nosh.")
    p.add_argument("--phase", choices=["both", "predict", "judge"], default="both",
                   help="'predict' only generates; 'judge' only scores existing predictions.")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already fully scored in the existing output CSV")
    p.add_argument("--nosh-temperature", type=float, default=0.0)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--nosh-max-tokens", type=int, default=128)
    p.add_argument("--judge-max-tokens", type=int, default=256)
    p.add_argument("--nosh-gpu-mem", type=float, default=0.85)
    p.add_argument("--judge-gpu-mem", type=float, default=0.85)
    p.add_argument("--nosh-max-len", type=int, default=4096)
    p.add_argument("--judge-max-len", type=int, default=6144)
    p.add_argument("--judge-quantization", default=None,
                   help="vLLM quantization for judge (e.g. 'bitsandbytes', 'awq', 'gptq'). "
                        "Leave unset to auto-detect from model config.")
    return p.parse_args()


def read_input_csv(in_path: Path) -> pd.DataFrame:
    """Read the input CSV and normalize column names.

    Accepts either (prompt, command) or (input_text, bash_command).
    """
    df = pd.read_csv(in_path)
    if "prompt" not in df.columns and "input_text" in df.columns:
        df = df.rename(columns={"input_text": "prompt"})
    if "command" not in df.columns and "bash_command" in df.columns:
        df = df.rename(columns={"bash_command": "command"})
    if "prompt" not in df.columns or "command" not in df.columns:
        sys.exit(f"input CSV must have columns (prompt, command) or (input_text, bash_command); got {list(df.columns)}")
    return df


def main():
    args = parse_args()

    in_path = DATA_DIR / args.file_name
    out_path = Path(args.output) if args.output else RESULTS_DIR / args.file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = out_path.with_suffix(".predictions.csv")

    df = read_input_csv(in_path)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Loaded {len(df)} rows from {in_path}")

    done_prompts: set[str] = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done_prompts = set(prev[prev["overall"].notna()]["prompt"].astype(str).tolist())
        print(f"Resume: skipping {len(done_prompts)} already-scored rows")
        df = df[~df["prompt"].astype(str).isin(done_prompts)].reset_index(drop=True)

    prompts = df["prompt"].astype(str).tolist()
    expecteds = df["command"].astype(str).str.strip().tolist()

    # ---------- Phase 1: predict ----------
    if args.phase in ("both", "predict"):
        if args.has_expected:
            print("--has-expected: using CSV `command` as the candidate, skipping nosh inference")
            commands = list(expecteds)
        elif len(prompts) == 0:
            commands = []
        else:
            print(f"\n[phase 1] loading nosh model: {args.nosh_model}")
            nosh = VLLMRunner(
                args.nosh_model,
                PROMPTS_DIR / "nosh.md",
                PROMPTS_DIR / "nosh_examples.json",
                max_model_len=args.nosh_max_len,
                gpu_memory_utilization=args.nosh_gpu_mem,
            )
            print(f"generating {len(prompts)} predictions...")
            commands = nosh.generate_batch(
                prompts,
                temperature=args.nosh_temperature,
                max_tokens=args.nosh_max_tokens,
            )
            nosh.shutdown()

        # Auto checks
        formattings = [check_formatting(c) for c in commands]
        syntaxes = [check_syntax(c) for c in commands]
        exact_matches = [
            normalize_for_compare(c) == normalize_for_compare(e) if e else False
            for c, e in zip(commands, expecteds)
        ]

        pred_df = pd.DataFrame({
            "prompt": prompts,
            "command": commands,
            "expected": expecteds,
            "exact_match": exact_matches,
            "formatting": formattings,
            "syntax_valid": syntaxes,
        })
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)
        print(f"wrote {len(pred_df)} predictions to {predictions_path}")

        if args.phase == "predict":
            return
    else:
        # phase == "judge": load predictions from intermediate file
        if not predictions_path.exists():
            sys.exit(f"--phase judge: predictions file not found at {predictions_path}")
        pred_df = pd.read_csv(predictions_path)
        # Restrict to rows we still need (in case of --resume)
        if done_prompts:
            pred_df = pred_df[~pred_df["prompt"].astype(str).isin(done_prompts)].reset_index(drop=True)
        print(f"loaded {len(pred_df)} predictions from {predictions_path}")

    # ---------- Phase 2: judge ----------
    if args.phase in ("both", "judge") and len(pred_df) > 0:
        print(f"\n[phase 2] loading judge model: {args.judge_model}")
        judge = VLLMRunner(
            args.judge_model,
            PROMPTS_DIR / "judge.md",
            PROMPTS_DIR / "judge_examples.json",
            max_model_len=args.judge_max_len,
            gpu_memory_utilization=args.judge_gpu_mem,
            quantization=args.judge_quantization,
        )

        judge_inputs = [
            json.dumps({
                "input_text": str(row["prompt"]),
                "command": str(row["command"]),
                "expected": str(row["expected"]),
                "auto": {
                    "formatting": bool(row["formatting"]),
                    "syntax_valid": bool(row["syntax_valid"]),
                },
            })
            for _, row in pred_df.iterrows()
        ]
        print(f"scoring {len(judge_inputs)} rows...")
        raws = judge.generate_batch(
            judge_inputs,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )
        judge.shutdown()

        results = []
        for (_, row), raw in zip(pred_df.iterrows(), raws):
            parsed = parse_judge_output(raw)
            r = {
                "prompt": str(row["prompt"]),
                "command": str(row["command"]),
                "expected": str(row["expected"]),
                "exact_match": bool(row["exact_match"]),
                "formatting": bool(row["formatting"]),
                "syntax_valid": bool(row["syntax_valid"]),
            }
            if parsed is not None:
                r.update({
                    "tool": bool(parsed["tool"]),
                    "flags": int(parsed["flags"]),
                    "args": int(parsed["args"]),
                    "constraints": int(parsed["constraints"]),
                    "safety": bool(parsed["safety"]),
                    "overall": int(parsed["overall"]),
                    "reasoning": str(parsed["reasoning"])[:300],
                })
            else:
                r.update({
                    "tool": None, "flags": None, "args": None, "constraints": None,
                    "safety": None, "overall": None, "reasoning": None,
                })
                print(f"\n[judge parse failed] prompt={r['prompt']!r}\n  raw={raw!r}\n",
                      file=sys.stderr)
            results.append(r)

        out_df = pd.DataFrame(results, columns=RUBRIC_COLUMNS)

        if args.resume and out_path.exists():
            prev = pd.read_csv(out_path)
            keep_prev = prev[~prev["prompt"].astype(str).isin(out_df["prompt"].astype(str))]
            out_df = pd.concat([keep_prev, out_df], ignore_index=True)

        out_df.to_csv(out_path, index=False)
        print(f"\nWrote {len(out_df)} rows to {out_path}")

        stats = aggregate(out_df)
        print("\n=== Aggregate ===")
        for k, v in stats.items():
            print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
