"""Interactive REPL for the finetuned nosh GGUF model.

Loads the system prompt from ./prompts/nosh.md and few-shot examples from
./prompts/nosh_examples.json, then lets you chat with the model. Each user
turn is sent independently (no history) so you're testing single-shot NL->bash
behavior — same shape as eval.py.

Meta-commands inside the REPL:
    /reload   re-read prompt + examples from disk (edit them without restart)
    /show     print the current system prompt + few-shots
    /temp X   change sampling temperature
    /quit     exit (Ctrl-D also works)

Run:
    python play.py
    python play.py --model ./checkpoints/nosh-fp16.gguf --temperature 0
"""

import argparse
import json
from pathlib import Path

from llama_cpp import Llama


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "checkpoints" / "nosh-v2-Q4_K_M.gguf"
DEFAULT_SYSTEM = ROOT / "prompts" / "nosh.md"
DEFAULT_EXAMPLES = ROOT / "prompts" / "nosh_examples.json"


def load_prompts(system_path: Path, examples_path: Path):
    with open(system_path) as f:
        system_prompt = f.read().strip()
    with open(examples_path) as f:
        fewshots = json.load(f)
    return system_prompt, fewshots


def build_messages(system_prompt: str, fewshots: list, user_input: str):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(fewshots)
    messages.append({"role": "user", "content": user_input})
    return messages


def show_context(system_prompt: str, fewshots: list):
    print("\n--- system ---")
    print(system_prompt)
    print("\n--- few-shots ---")
    for m in fewshots:
        print(f"  [{m['role']}] {m['content']}")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM))
    p.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="-1 = offload all layers to GPU, 0 = CPU only")
    return p.parse_args()


def main():
    args = parse_args()

    system_path = Path(args.system_prompt)
    examples_path = Path(args.examples)
    system_prompt, fewshots = load_prompts(system_path, examples_path)

    print(f"Loading {args.model} ...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        chat_format="chatml",
        verbose=False,
    )
    print("Ready. Type a request, /show, /reload, /temp N, or /quit.\n")
    show_context(system_prompt, fewshots)

    temperature = args.temperature
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/show":
            show_context(system_prompt, fewshots)
            continue
        if user_input == "/reload":
            system_prompt, fewshots = load_prompts(system_path, examples_path)
            print(f"reloaded ({len(fewshots)} few-shot messages)")
            continue
        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"temperature = {temperature}")
                except ValueError:
                    print("usage: /temp <float>")
            else:
                print(f"current temperature = {temperature}")
            continue

        messages = build_messages(system_prompt, fewshots, user_input)
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=args.max_tokens,
        )
        reply = out["choices"][0]["message"]["content"].strip()
        print(reply)


if __name__ == "__main__":
    main()
