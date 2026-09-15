"""Benchmark GGUF quant tiers x few-shot counts for the final nosh CLI.

Predicts with llama-cpp-python (the real deployment path, n_ctx=512) for each
(quant, k_fewshot) config, then judges everything in a single vLLM pass with the
14B judge, and prints per-config aggregates.

Config syntax:  <QUANT>:<K>   e.g.  Q4_K_M:0  Q4_K_M:2  Q3_K_M:2

Run:
    python bench_quant_fewshot.py --rows 250 --configs Q4_K_M:0 Q4_K_M:2 Q4_K_M:4 Q4_K_M:8
"""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

# reuse the eval harness's judge + checks (importing pulls in vllm, that's fine)
import eval as ev

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
PROMPTS = ROOT / "prompts"


def load_nosh_prompt(k_fewshot: int):
    system = (PROMPTS / "nosh.md").read_text().strip()
    allfs = json.loads((PROMPTS / "nosh_examples.json").read_text())
    fewshots = allfs[: 2 * k_fewshot]  # k pairs = 2k messages
    return system, fewshots


def predict_gguf(quant: str, k_fewshot: int, prompts, n_ctx=512, max_tokens=128):
    from llama_cpp import Llama
    model_path = str(CKPT / f"nosh-v2-{quant}.gguf")
    system, fewshots = load_nosh_prompt(k_fewshot)
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1,
                chat_format="chatml", verbose=False)
    out = []
    for p in prompts:
        msgs = [{"role": "system", "content": system}] + fewshots + [{"role": "user", "content": p}]
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=max_tokens)
        out.append(r["choices"][0]["message"]["content"].strip())
    del llm
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="test_6k.csv")
    ap.add_argument("--rows", type=int, default=250)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--out", default="results/bench_quant_fewshot.csv")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "datasets" / args.test_file).head(args.rows)
    prompts = df["prompt"].astype(str).tolist()
    expecteds = df["command"].astype(str).str.strip().tolist()
    print(f"benchmark on {len(prompts)} rows from {args.test_file}, n_ctx={args.n_ctx}")

    # ---- phase 1: predict every config with llama-cpp ----
    all_rows = []
    for cfg in args.configs:
        quant, k = cfg.split(":")
        k = int(k)
        print(f"\n[predict] {quant} k={k} ...")
        cmds = predict_gguf(quant, k, prompts, n_ctx=args.n_ctx)
        for p, c, e in zip(prompts, cmds, expecteds):
            all_rows.append({
                "config": cfg, "quant": quant, "k": k,
                "prompt": p, "command": c, "expected": e,
                "exact_match": ev.normalize_for_compare(c) == ev.normalize_for_compare(e) if e else False,
                "formatting": ev.check_formatting(c),
                "syntax_valid": ev.check_syntax(c),
            })

    pred = pd.DataFrame(all_rows)

    # ---- phase 2: judge everything in one vLLM pass ----
    print(f"\n[judge] scoring {len(pred)} rows with {ev.__name__} judge ...")
    judge = ev.VLLMRunner(
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        PROMPTS / "judge.md", PROMPTS / "judge_examples.json",
        max_model_len=6144, gpu_memory_utilization=0.85,
    )
    judge_inputs = [
        json.dumps({
            "input_text": r["prompt"], "command": r["command"], "expected": r["expected"],
            "auto": {"formatting": bool(r["formatting"]), "syntax_valid": bool(r["syntax_valid"])},
        })
        for _, r in pred.iterrows()
    ]
    raws = judge.generate_batch(judge_inputs, temperature=0.0, max_tokens=256)
    judge.shutdown()

    overalls, ge4 = [], []
    for raw in raws:
        parsed = ev.parse_judge_output(raw)
        overalls.append(int(parsed["overall"]) if parsed else None)
    pred["overall"] = overalls

    pred.to_csv(ROOT / args.out, index=False)

    # ---- aggregate per config ----
    print("\n==================== RESULTS ====================")
    rows = []
    sizes = {q: (CKPT / f"nosh-v2-{q}.gguf").stat().st_size / 1048576
             for q in pred["quant"].unique()}
    for cfg, g in pred.groupby("config", sort=False):
        j = g[g["overall"].notna()]
        rows.append({
            "config": cfg,
            "size_MB": round(sizes[g["quant"].iloc[0]]),
            "overall": round(j["overall"].mean(), 3) if len(j) else None,
            "pct>=4": round((j["overall"] >= 4).mean(), 3) if len(j) else None,
            "exact": round(g["exact_match"].mean(), 3),
            "syntax": round(g["syntax_valid"].mean(), 3),
            "fmt": round(g["formatting"].mean(), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
