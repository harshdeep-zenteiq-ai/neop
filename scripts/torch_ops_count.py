import json
from collections import defaultdict

with open("torch_trace.json") as f:
    trace = json.load(f)

events = trace["traceEvents"]

dtype_counts = defaultdict(int)
total_ops = 0

for e in events:
    if e.get("ph") != "X":          # only duration events (actual ops)
        continue
    if e.get("cat") not in ("cpu_op", "cuda_runtime", "kernel"):
        continue

    total_ops += 1

    # dtype lives in args dict under various keys
    args = e.get("args", {})
    dtypes = args.get("Input type", args.get("dtype", args.get("scalar_type", "")))

    if isinstance(dtypes, list):
        for d in dtypes:
            if d:
                dtype_counts[str(d)] += 1
    elif dtypes:
        dtype_counts[str(dtypes)] += 1

print(f"Total ops: {total_ops}")
print("\nDtype breakdown:")
for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
    print(f"  {dtype}: {count}")

