#!/usr/bin/env python
"""nosh — natural language to a bash command.

Cold-loads the finetuned Qwen2.5-Coder-1.5B GGUF on every call (no daemon),
prints ONLY the bash command on stdout so a shell wrapper can drop it onto the
next prompt line.

    nosh list every python file changed in the last day

Tunables via env:
    NOSH_MODEL        path to the .gguf            (default: nosh-v2-Q4_K_M.gguf)
    NOSH_FEWSHOT      number of few-shot pairs     (default: see N_FEWSHOT)
    NOSH_GPU_LAYERS   layers to offload to GPU     (default: -1 = all if a GPU
                      build is installed; harmless no-op on CPU-only wheels)
    NOSH_CTX          context window               (default: 512)
"""

import contextlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def quiet_stderr():
    """Silence llama.cpp's C-level stderr chatter (load logs, ctx warnings).

    Redirects fd 2 to /dev/null at the OS level so the wrapper's terminal stays
    clean. Our own error messages are printed before/after this block.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

# --- defaults chosen from bench_quant_fewshot.py ---
DEFAULT_MODEL = ROOT / "checkpoints" / "nosh-v2-Q4_K_M.gguf"
N_FEWSHOT = 2
N_CTX = 512
# ---------------------------------------------------

SYSTEM_PATH = ROOT / "prompts" / "nosh.md"
EXAMPLES_PATH = ROOT / "prompts" / "nosh_examples.json"


def build_messages(query: str):
    system = SYSTEM_PATH.read_text().strip()
    k = int(os.environ.get("NOSH_FEWSHOT", N_FEWSHOT))
    fewshots = json.loads(EXAMPLES_PATH.read_text())[: 2 * k]
    return [{"role": "system", "content": system}] + fewshots + [
        {"role": "user", "content": query}
    ]


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("usage: nosh <natural language>", file=sys.stderr)
        sys.exit(1)

    # import here so --help / usage errors don't pay the import cost
    from llama_cpp import Llama

    model = os.environ.get("NOSH_MODEL", str(DEFAULT_MODEL))
    if not Path(model).exists():
        print(f"nosh: model not found: {model}", file=sys.stderr)
        sys.exit(2)

    with quiet_stderr():
        llm = Llama(
            model_path=model,
            n_ctx=int(os.environ.get("NOSH_CTX", N_CTX)),
            n_gpu_layers=int(os.environ.get("NOSH_GPU_LAYERS", -1)),
            chat_format="chatml",
            verbose=False,
        )
        out = llm.create_chat_completion(
            messages=build_messages(query),
            temperature=0.0,
            max_tokens=128,
        )
    command = out["choices"][0]["message"]["content"].strip()
    # stdout = the command only; nothing else, so a wrapper can consume it cleanly
    print(command)


if __name__ == "__main__":
    main()
