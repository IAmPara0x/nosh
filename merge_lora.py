"""Merge the trained LoRA adapter into the base model and save in 16-bit.

Reads the base model path from the adapter's adapter_config.json so we don't
hardcode it. Output is a standalone HF-format model directory that can be
fed to llama.cpp's convert_hf_to_gguf.py for GGUF conversion.
"""

import unsloth  # noqa: F401  must precede transformers/trl
from unsloth import FastLanguageModel

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="./checkpoints/qwen2.5-coder-1.5b-nosh")
    p.add_argument("--output-dir", default="./checkpoints/qwen2.5-coder-1.5b-nosh-merged")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--save-method", choices=["merged_16bit", "merged_4bit"],
                   default="merged_16bit")
    return p.parse_args()


def main():
    args = parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)
    base_model_path = cfg["base_model_name_or_path"]
    print(f"Base model : {base_model_path}")
    print(f"Adapter dir: {adapter_dir}")

    # Passing the adapter dir to from_pretrained makes Unsloth resolve the base
    # model from adapter_config.json and load the LoRA on top in one shot. This
    # is what save_pretrained_merged expects (load_adapter() afterwards only
    # registers a named adapter and is not what gets merged).
    print("Loading base model + LoRA adapter in 16-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        do_one_hot_enc=True
    )

    print(f"Saving merged model ({args.save_method}) to {args.output_dir}")
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method=args.save_method,
    )
    print("Done.")


if __name__ == "__main__":
    main()
