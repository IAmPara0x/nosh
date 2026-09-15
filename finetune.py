"""Finetune Qwen2.5-Coder-1.5B-Instruct on the nosh NL->bash dataset.

Train data = ./datasets/cleaned/000.csv ... 008.csv + a balanced subset of
nl2bash.csv. 009.csv and 010.csv are reserved for evaluation (eval.py).

Run:
    python finetune.py
    python finetune.py --epochs 5 --rank 64 --per-tool-cap 30
"""

import unsloth  # noqa: F401  must be imported before transformers/trl
from unsloth import FastLanguageModel

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
import random


ROOT = Path(__file__).resolve().parent
CLEANED_DIR = ROOT / "datasets" / "cleaned"
PROMPTS_DIR = ROOT / "prompts"

TEST_FILES = {"009.csv", "010.csv"}
NL2BASH_FILE = "nl2bash.csv"


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a 2-column (input_text, bash_command) CSV tolerantly.

    Some rows have unquoted commas in input_text (e.g. ``cores 0,1``) so naive
    parsing splits them into 3+ fields. We use the csv stdlib (which is more
    forgiving than pandas) and re-join the leading fields into input_text.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for r in reader:
            if len(r) < 2:
                continue
            # Last field is bash_command; everything before it forms input_text
            input_text = ",".join(r[:-1])
            bash_command = r[-1]
            rows.append((input_text, bash_command))
    return pd.DataFrame(rows, columns=["input_text", "bash_command"])


def load_train_csv(path: Path) -> pd.DataFrame:
    """Load a pre-built training CSV (prepare_data.py output).

    Accepts (prompt, command) or (input_text, bash_command) and normalizes to
    the (input_text, bash_command) schema the rest of this script expects.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "input_text" not in df.columns and "prompt" in df.columns:
        df = df.rename(columns={"prompt": "input_text"})
    if "bash_command" not in df.columns and "command" in df.columns:
        df = df.rename(columns={"command": "bash_command"})
    df = (df[["input_text", "bash_command"]]
          .dropna()
          .drop_duplicates()
          .reset_index(drop=True))
    print(f"  loaded pre-built train set: {len(df)} rows from {path}")
    return df


def load_training_data(per_tool_cap: int, nl2bash_total_cap: int, seed: int):
    """Combine cleaned 000-008 with a balanced subset of nl2bash.

    nl2bash is dominated by `find` (~60%). We cap each tool (first whitespace
    token of bash_command) at `per_tool_cap`, then optionally trim to
    `nl2bash_total_cap` overall.
    """
    main_frames = []
    for path in sorted(CLEANED_DIR.glob("*.csv")):
        if path.name in TEST_FILES or path.name == NL2BASH_FILE:
            continue
        main_frames.append(read_csv_robust(path))
    main_df = pd.concat(main_frames, ignore_index=True)
    print(f"  cleaned 000-008: {len(main_df)} rows from {len(main_frames)} files")

    nl2bash = read_csv_robust(CLEANED_DIR / NL2BASH_FILE)
    tools = nl2bash["bash_command"].fillna("").str.strip().str.split().str[0]
    sampled = (
        nl2bash.groupby(tools, group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_tool_cap), random_state=seed))
               .reset_index(drop=True)
    )
    if len(sampled) > nl2bash_total_cap:
        sampled = sampled.sample(nl2bash_total_cap, random_state=seed).reset_index(drop=True)
    print(f"  nl2bash sampled: {len(sampled)} rows "
          f"(cap/tool={per_tool_cap}, total cap={nl2bash_total_cap})")

    full = pd.concat([main_df, sampled], ignore_index=True)
    full = (full.dropna(subset=["input_text", "bash_command"])
                .drop_duplicates(subset=["input_text", "bash_command"])
                .reset_index(drop=True))
    full = full.sample(frac=1).reset_index(drop=True)
    print(f"  combined train set: {len(full)} rows")
    return full


def load_prompt_assets():
    with open(PROMPTS_DIR / "nosh.md") as f:
        system_prompt = f.read().strip()
    with open(PROMPTS_DIR / "nosh_examples.json") as f:
        fewshots = json.load(f)
    return system_prompt, fewshots


class SamplePreviewCallback(TrainerCallback):
    """Generate predictions for a fixed set of prompts every N steps."""

    def __init__(self, model, tokenizer, rows, system_prompt, fewshots,
                 every_n_steps: int, sample_count: int, max_new_tokens: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.rows = rows  # list of dicts with input_text + bash_command
        self.system_prompt = system_prompt
        self.fewshots = fewshots
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.sample_count = sample_count

    def _generate(self, input_text: str) -> str:
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.extend(self.fewshots)
        conversation.append({"role": "user", "content": input_text})
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                max_length=None
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _print_samples(self, step: int):
        was_training = self.model.training
        self.model.eval()
        try:
            print(f"\n=== sample previews @ step {step} ===")
            sampled_rows = random.sample(self.rows, self.sample_count)
            for row in sampled_rows:
                pred = self._generate(row["input_text"])
                ok = "OK " if pred == row["bash_command"] else "    "
                print(f"  {ok}input    : {row['input_text']}")
                print(f"      expected : {row['bash_command']}")
                print(f"      predicted: {pred}")
            print("=" * 40)
        finally:
            if was_training:
                self.model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0:
            return
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._print_samples(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self._print_samples(state.global_step)


def make_sample(row, system_prompt, fewshots, tokenizer):
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(fewshots)
    conversation.append({"role": "user", "content": str(row["input_text"])})
    completion = [{"role": "assistant", "content": str(row["bash_command"])}]

    prompt_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    completion_text = tokenizer.apply_chat_template(
        completion, tokenize=False
    )
    return {"prompt": prompt_text, "completion": completion_text}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/paradox/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a")
    p.add_argument("--output-dir", default="./checkpoints/qwen2.5-coder-1.5b-nosh")
    p.add_argument("--train-csv", default=None,
                   help="Pre-built training CSV (prepare_data.py output). If set, "
                        "overrides the cleaned+nl2bash assembly.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--per-tool-cap", type=int, default=50,
                   help="Max samples per first-token tool in nl2bash")
    p.add_argument("--nl2bash-total-cap", type=int, default=2000,
                   help="Hard cap on total nl2bash samples after per-tool capping")
    p.add_argument("--quantization", choices=["8bit", "4bit", "none"], default="8bit")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save-merged", action="store_true",
                   help="Also save a merged-16bit copy for plug-in use with eval.py")
    p.add_argument("--no-padding-free", action="store_true",
                   help="Disable padding_free (needed on <17GB VRAM)")
    p.add_argument("--sample-every", type=int, default=50,
                   help="Print model predictions every N steps (0 disables)")
    p.add_argument("--sample-count", type=int, default=5,
                   help="Number of preview samples to generate each interval")
    p.add_argument("--sample-source", choices=["train", "test"], default="test",
                   help="Pull preview prompts from train data or held-out 009/010")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading training data...")
    if args.train_csv:
        train_df = load_train_csv(Path(args.train_csv))
    else:
        train_df = load_training_data(
            per_tool_cap=args.per_tool_cap,
            nl2bash_total_cap=args.nl2bash_total_cap,
            seed=args.seed,
        )

    system_prompt, fewshots = load_prompt_assets()

    print(f"Loading model: {args.model} ({args.quantization})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=(args.quantization == "4bit"),
        load_in_8bit=(args.quantization == "8bit"),
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),  # set HF_TOKEN in env for gated model downloads
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print("Building prompts...")
    samples = [make_sample(row, system_prompt, fewshots, tokenizer)
               for _, row in train_df.iterrows()]
    train_dataset = Dataset.from_pandas(pd.DataFrame(samples))
    print(f"  train_dataset: {len(train_dataset)} examples")

    preview_rows = []
    if args.sample_every > 0:
        if args.sample_source == "test":
            test_frames = [read_csv_robust(CLEANED_DIR / name) for name in sorted(TEST_FILES)]
            pool = pd.concat(test_frames, ignore_index=True)
        else:
            pool = train_df
        preview_rows = (pool.to_dict(orient="records"))
        print(f"Sample previews every {args.sample_every} steps "
              f"({len(preview_rows)} rows from {args.sample_source})")

    callbacks = []
    if preview_rows:
        callbacks.append(SamplePreviewCallback(
            model=model,
            tokenizer=tokenizer,
            rows=preview_rows,
            system_prompt=system_prompt,
            fewshots=fewshots,
            every_n_steps=args.sample_every,
            sample_count=args.sample_count
        ))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            padding_free=not args.no_padding_free,
            completion_only_loss=True,
            save_strategy="epoch",
            save_total_limit=2,
        ),
    )

    print("Training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged:
        merged_dir = f"{args.output_dir}-merged"
        print(f"Saving merged-16bit model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("Done.")


if __name__ == "__main__":
    main()
