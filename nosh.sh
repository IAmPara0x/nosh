# nosh shell integration — source this from ~/.zshrc or ~/.bashrc:
#     source /path/to/nosh/nosh.sh
#
# Usage:   nosh list every python file changed today   <Enter>
# The generated command appears, pre-typed, on your NEXT prompt line; just press
# Enter to run it (or edit it first).
#
# Overrides (optional):
#   NOSH_PYTHON   python interpreter to use   (default: python3)
#   NOSH_MODEL    path to a .gguf             (default: <nosh>/checkpoints/nosh-v2-Q4_K_M.gguf)

# Resolve this script's own directory so nosh works wherever it is cloned.
if [ -n "$ZSH_VERSION" ]; then
  NOSH_DIR="${${(%):-%x}:A:h}"
elif [ -n "$BASH_VERSION" ]; then
  NOSH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
NOSH_PY="${NOSH_PYTHON:-python3}"

if [ -n "$ZSH_VERSION" ]; then
  # zsh: print -z pushes text onto the editor buffer stack -> next prompt is pre-filled.
  nosh() {
    if [ "$#" -eq 0 ]; then echo "usage: nosh <natural language>" >&2; return 1; fi
    local cmd
    cmd="$("$NOSH_PY" "$NOSH_DIR/nosh.py" "$@")" || return $?
    [ -n "$cmd" ] && print -z -- "$cmd"
  }
elif [ -n "$BASH_VERSION" ]; then
  # bash has no clean equivalent of zsh's `print -z`: pre-filling the prompt
  # buffer is only possible from inside a readline keybinding. So we print the
  # command and stage it into history -> press Up then Enter to run it.
  nosh() {
    if [ "$#" -eq 0 ]; then echo "usage: nosh <natural language>" >&2; return 1; fi
    local cmd
    cmd="$("$NOSH_PY" "$NOSH_DIR/nosh.py" "$@")" || return $?
    [ -z "$cmd" ] && return 0
    history -s "$cmd"
    printf '%s\n' "$cmd"
  }
fi
