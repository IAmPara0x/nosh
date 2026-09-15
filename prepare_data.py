"""Build clean train/test splits for the nosh NL->bash finetune.

Inputs (all under ./datasets/cleaned):
  000.csv .. 008.csv      curated devops-heavy pairs (train)
  009.csv, 010.csv        curated held-out test (never trained on)
  nl2bash.csv             large find-dominated corpus (balanced subset -> train)
  bash_command_data_6k    large, complex, well-formed corpus (split train/test)

Outputs (under ./datasets):
  train.csv               combined, de-duplicated, comment-stripped training set
  test_6k.csv             held-out slice of the 6k corpus (complex eval)

The 6k corpus stores some commands with trailing "# [output ...]" annotation
lines. Those would teach the model to emit prose comments, which violates the
"output ONLY bash" contract, so we strip pure-comment lines (keeping real
`#!`-shebang scripts intact).
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEANED = ROOT / "datasets" / "cleaned"
OUT_DIR = ROOT / "datasets"

TEST_FILES = {"009.csv", "010.csv"}


def strip_comment_lines(cmd: str) -> str:
    """Drop blank / pure-comment lines from a command, unless it's a #! script."""
    cmd = str(cmd)
    if cmd.strip().startswith("#!"):
        return cmd.strip()  # real script: keep as-is
    lines = []
    for line in cmd.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue  # annotation / example-output comment
        lines.append(line)
    return "\n".join(lines).strip()


def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # normalize to (prompt, command)
    if "prompt" not in df.columns and "input_text" in df.columns:
        df = df.rename(columns={"input_text": "prompt"})
    if "command" not in df.columns and "bash_command" in df.columns:
        df = df.rename(columns={"bash_command": "command"})
    return df[["prompt", "command"]].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["prompt", "command"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["command"] = df["command"].astype(str).map(strip_comment_lines)
    df = df[(df["prompt"] != "") & (df["command"] != "")]
    return df.drop_duplicates(subset=["prompt", "command"]).reset_index(drop=True)


def balanced_nl2bash(df: pd.DataFrame, per_tool_cap: int, total_cap: int, seed: int) -> pd.DataFrame:
    tools = df["command"].str.split().str[0]
    sampled = (
        df.groupby(tools, group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_tool_cap), random_state=seed))
          .reset_index(drop=True)
    )
    if len(sampled) > total_cap:
        sampled = sampled.sample(total_cap, random_state=seed).reset_index(drop=True)
    return sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-6k", type=int, default=600, help="rows to hold out from the 6k corpus")
    p.add_argument("--nl2bash-per-tool", type=int, default=30)
    p.add_argument("--nl2bash-total", type=int, default=1800)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    # ---- main curated train (000-008) ----
    main_files = [f for f in sorted(glob.glob(str(CLEANED / "0[01][0-9].csv")))
                  if Path(f).name not in TEST_FILES]
    main_df = clean(pd.concat([load_pairs(Path(f)) for f in main_files], ignore_index=True))
    print(f"cleaned 000-008: {len(main_df)} rows from {len(main_files)} files")

    # ---- 6k: split into train/test ----
    six = clean(load_pairs(CLEANED / "bash_command_data_6k.csv"))
    six = six.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_6k = six.head(args.test_6k).reset_index(drop=True)
    train_6k = six.tail(len(six) - args.test_6k).reset_index(drop=True)
    print(f"6k: {len(train_6k)} train / {len(test_6k)} test")

    # ---- nl2bash: balanced subset ----
    nl = clean(load_pairs(CLEANED / "nl2bash.csv"))
    nl_sub = balanced_nl2bash(nl, args.nl2bash_per_tool, args.nl2bash_total, args.seed)
    print(f"nl2bash: {len(nl_sub)} balanced (per-tool={args.nl2bash_per_tool}, cap={args.nl2bash_total})")

    # ---- combine ----
    train = pd.concat([main_df, train_6k, nl_sub], ignore_index=True)
    train = train.drop_duplicates(subset=["prompt", "command"]).reset_index(drop=True)

    # ---- leakage guard: drop any train prompt that appears in a test set ----
    held = load_pairs(CLEANED / "009.csv")
    held = pd.concat([held, load_pairs(CLEANED / "010.csv")], ignore_index=True)
    test_prompts = set(held["prompt"].astype(str).str.strip()) | set(test_6k["prompt"])
    before = len(train)
    train = train[~train["prompt"].isin(test_prompts)].reset_index(drop=True)
    print(f"leakage guard: dropped {before - len(train)} train rows overlapping test prompts")

    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_path = OUT_DIR / "train.csv"
    test_path = OUT_DIR / "test_6k.csv"
    train.to_csv(train_path, index=False)
    test_6k.to_csv(test_path, index=False)
    print(f"\nwrote {len(train)} -> {train_path}")
    print(f"wrote {len(test_6k)} -> {test_path}")

    # quick profile
    tool = train["command"].str.split().str[0]
    print(f"\ntrain uniq tools: {tool.nunique()}, avg cmd len: {train['command'].str.len().mean():.1f}")
    print("top tools:", tool.value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
