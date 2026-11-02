"""Regenerate the results section of README.md from results/*.json.

Usage:  python tools/update_readme.py
Only the text between the RESULTS:START / RESULTS:END markers is replaced.
Uses the standard library only, so it runs on any machine (no GPU needed).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README = os.path.join(ROOT, "README.md")
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def num(value, digits=1):
    return "-" if value is None else f"{value:,.{digits}f}"


def gpu_section(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    peak = data.get("peak_bandwidth_gbs")
    lines = [f"#### {data['gpu']} (compute capability {data.get('compute_capability', '?')})", ""]

    stem = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(os.path.join(ROOT, "docs", f"bandwidth_{stem}.png")):
        lines += [f"![Memory bandwidth on {data['gpu']}](docs/bandwidth_{stem}.png)", ""]

    lines += [
        "| Kernel | Time (ms) | Bandwidth (GB/s) | % of peak | GFLOP/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in data["results"]:
        bw = r.get("bandwidth_gbs")
        pct = "-" if (bw is None or not peak) else f"{bw / peak * 100:.0f}%"
        lines.append(
            f"| `{r['kernel']}` | {num(r['time_ms'], 3)} | {num(bw)} | {pct} | {num(r.get('gflops'))} |"
        )

    notes = []
    if peak:
        notes.append(f"Peak memory bandwidth: {peak:g} GB/s.")
    if data.get("torch"):
        notes.append(f"PyTorch {data['torch']}, CUDA {data.get('cuda')}.")
    if notes:
        lines += ["", " ".join(notes)]
    return "\n".join(lines)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "results", "*.json")))
    if not files:
        sys.exit("No results/*.json files found. Run: python bench.py --save")

    block = START + "\n\n" + "\n\n".join(gpu_section(p) for p in files) + "\n\n" + END

    with open(README, encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"Markers {START} ... {END} not found in README.md")
    with open(README, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda m: block, text))
    print(f"Updated README.md from {len(files)} results file(s).")


if __name__ == "__main__":
    main()
