# Installing nosh

`nosh` turns a natural-language request into a Bash command and drops it onto
your next shell prompt. It runs a 1.5B-parameter fine-tuned model **locally** —
no GPU and no network required at inference time. Peak RAM ≈ 1.7 GB, latency
≈ 1.4–1.6 s per query on a laptop CPU.

- **Model weights:** https://huggingface.co/I-Am-Paradox/nosh-qwen2.5-coder-1.5b
- **Works on:** Linux, macOS, Windows (WSL / Git Bash)

---

## 1. Prerequisites

- **Python 3.9+** (`python3 --version`)
- A C compiler toolchain for building `llama-cpp-python`:
  - **Linux:** `sudo pacman -S base-devel` (Arch) / `sudo apt install build-essential` (Debian/Ubuntu)
  - **macOS:** `xcode-select --install`
  - **Windows:** use **WSL** (recommended) or install “Desktop development with C++” from the Visual Studio Build Tools

---

## 2. Get the nosh code

```sh
git clone https://github.com/IAmParadox/nosh.git
cd nosh
```

> No GitHub remote yet? Copy these four files from the project instead and keep
> the layout: `nosh.py`, `nosh.sh`, `prompts/nosh.md`, `prompts/nosh_examples.json`.

## 3. Install the inference runtime

A virtual environment keeps this isolated (optional but recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install llama-cpp-python
```

CPU inference works out of the box. To offload to a GPU (optional), build with a
backend instead, e.g. CUDA:

```sh
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## 4. Download the model weights (≈ 940 MB)

Into `checkpoints/` where `nosh.py` looks by default:

```sh
pip install -U "huggingface_hub[cli]"
hf download I-Am-Paradox/nosh-qwen2.5-coder-1.5b nosh-v2-Q4_K_M.gguf \
    --local-dir ./checkpoints
```

Or with plain `curl`:

```sh
mkdir -p checkpoints
curl -L -o checkpoints/nosh-v2-Q4_K_M.gguf \
  https://huggingface.co/I-Am-Paradox/nosh-qwen2.5-coder-1.5b/resolve/main/nosh-v2-Q4_K_M.gguf
```

## 5. Quick test

```sh
python3 nosh.py "list every python file changed in the last day"
# -> find . -name '*.py' -mtime -1
```

## 6. Enable the `nosh` shell command

Source the integration script from your shell rc file so a generated command
appears **pre-typed on your next prompt line** — just press Enter to run it.

**zsh** (`~/.zshrc`):
```sh
echo "source $(pwd)/nosh.sh" >> ~/.zshrc
source ~/.zshrc
```

**bash** (`~/.bashrc`):
```sh
echo "source $(pwd)/nosh.sh" >> ~/.bashrc
source ~/.bashrc
```

If you used a virtualenv, point nosh at that Python by adding, before the
`source` line above:

```sh
export NOSH_PYTHON="$(pwd)/.venv/bin/python"
```

### Usage

```
$ nosh count unique IP addresses in access.log sorted by frequency
$ awk '{print $1}' access.log | sort | uniq -c | sort -nr      # <- pre-filled, press Enter
```

- **zsh** pre-fills the next prompt (`print -z`).
- **bash** prints the command and stages it into history — press **Up**, then Enter.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `NOSH_MODEL` | `checkpoints/nosh-v2-Q4_K_M.gguf` | path to the `.gguf` weights |
| `NOSH_PYTHON` | `python3` | interpreter the shell wrapper calls |
| `NOSH_FEWSHOT` | `2` | number of in-context example pairs |
| `NOSH_GPU_LAYERS` | `-1` | layers to offload to GPU (`0` = CPU only) |
| `NOSH_CTX` | `512` | context window size |

## Troubleshooting

- **`model not found`** — the `.gguf` isn't at `checkpoints/nosh-v2-Q4_K_M.gguf`; re-check step 4 or set `NOSH_MODEL`.
- **`nosh: command not found`** — open a new shell or re-`source` your rc file; confirm `nosh.sh` was sourced.
- **`llama-cpp-python` build fails** — install the C toolchain in step 1; on Windows prefer WSL.
- **First call is slow** — the model cold-loads on every invocation by design (no daemon); the OS page cache makes subsequent calls faster.

## License & attribution

Fine-tuned from [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
(Apache-2.0) with LoRA. See the model card for evaluation details.
