"""
Reads train.txt and test.txt neighbor log files and prints overall
min / max / mean for gno_in and gno_out across all samples.

Usage:
    python summarize_neighbors.py --log_dir <path_to_log_dir>
"""

import argparse
import re
from pathlib import Path


def parse_log(path: Path):
    """Return lists of (total, min, max, mean) per sample for gno_in and gno_out."""
    gno_in  = []
    gno_out = []

    pattern = re.compile(
        r"gno_(in|out)\s+—\s+total:\s*(\d+)\s+min:\s*(\d+)\s+max:\s*(\d+)\s+mean:\s*([\d.]+)"
    )

    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                kind, total, mn, mx, mean = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))
                record = {"total": total, "min": mn, "max": mx, "mean": mean}
                if kind == "in":
                    gno_in.append(record)
                else:
                    gno_out.append(record)

    return gno_in, gno_out


def summarize(records, label):
    if not records:
        print(f"  {label}: no data found")
        return
    totals = [r["total"] for r in records]
    mins   = [r["min"]   for r in records]
    maxs   = [r["max"]   for r in records]
    means  = [r["mean"]  for r in records]
    # overall mean = weighted average using per-sample totals as proxy for n_query_points
    # (each sample's mean * n_query_points isn't stored, so we average the per-sample means)
    print(f"  {label} ({len(records)} samples):")
    print(f"    total neighbors — min: {min(totals)}  max: {max(totals)}  mean: {sum(totals)/len(totals):.1f}")
    print(f"    per-query min   — min: {min(mins)}  max: {max(mins)}  mean: {sum(mins)/len(mins):.2f}")
    print(f"    per-query max   — min: {min(maxs)}  max: {max(maxs)}  mean: {sum(maxs)/len(maxs):.2f}")
    print(f"    per-query mean  — min: {min(means):.2f}  max: {max(means):.2f}  mean: {sum(means)/len(means):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="neighbor_logs",
                        help="Directory containing train.txt and test.txt")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    for split in ("train", "test"):
        path = log_dir / f"{split}.txt"
        if not path.exists():
            print(f"\n[{split}] {path} not found, skipping.")
            continue

        gno_in, gno_out = parse_log(path)
        print(f"\n{'='*50}")
        print(f"  {split.upper()}  ({path})")
        print(f"{'='*50}")
        summarize(gno_in,  "gno_in")
        print()
        summarize(gno_out, "gno_out")

    print()


if __name__ == "__main__":
    main()
