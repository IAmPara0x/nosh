---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
  - bash
  - shell
  - code-generation
  - text-generation
  - gguf
  - llama-cpp
language:
  - en
pipeline_tag: text-generation
library_name: gguf
---

# nosh — Natural Language → Bash (Qwen2.5-Coder-1.5B, fine-tuned)

`nosh` converts a natural-language request into an executable Bash command. It is
a LoRA fine-tune of **Qwen2.5-Coder-1.5B-Instruct**, quantized to GGUF `Q4_K_M`
(≈ 940 MB) so it runs locally on a laptop CPU (peak RAM ≈ 1.7 GB, ≈ 1.4–1.6 s
per query).

> `nosh list every python file changed in the last day`
> → `find . -name '*.py' -mtime -1`

## Files

| File | Size | Use |
|---|---|---|
| `nosh-v2-Q4_K_M.gguf` | ~940 MB | 4-bit deployment model for `llama.cpp` / `llama-cpp-python` |

## Usage

```python
from llama_cpp import Llama

llm = Llama(model_path="nosh-v2-Q4_K_M.gguf", n_ctx=512, chat_format="chatml", verbose=False)
system = ("You are a Linux Bash CLI expert that converts a user's intent into a "
          "bash command. Output ONLY a valid bash command, no markdown or prose.")
out = llm.create_chat_completion(
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": "compress this folder to a gzip tarball"}],
    temperature=0.0, max_tokens=128,
)
print(out["choices"][0]["message"]["content"].strip())
```

The full CLI + shell integration (command lands pre-typed on your next prompt)
is in the project repository; see its `INSTALL.md`.

## Training

- **Base:** Qwen2.5-Coder-1.5B-Instruct
- **Method:** LoRA (rank 64, α 64) on attention + MLP projections, 2 epochs, effective batch 32, lr 2e-4 cosine.
- **Data:** 9,086 deduplicated NL→Bash pairs (890 distinct leading commands) mixing curated developer/sysadmin pairs, a complex multi-tool command corpus, and a tool-balanced NL2Bash subset. Comment/annotation lines were stripped; evaluation prompts were held out.

## Evaluation

Reference-aware **LLM-as-judge** (Qwen3-14B) scoring on a 600-example held-out
set of complex commands (`overall` is 1–5; `pct≥4` is the fraction rated ≥4):

| Model | overall | pct≥4 | exact | syntax |
|---|---|---|---|---|
| Base Qwen2.5-Coder-1.5B | 3.50 | 0.53 | 0.26 | 0.97 |
| **nosh (this model)** | **4.18** | **0.74** | **0.46** | **0.99** |

Fine-tuning on complex commands raised the mean judge score by **+0.68** and
nearly doubled exact-match accuracy over the base model.

## Limitations

- Judged by an LLM, not by execution; always review a generated command before running it.
- English input only; targets a Linux/GNU-coreutils environment.
- Best on the command styles seen in training; very unusual tools may be approximated.
