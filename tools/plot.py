"""Draw a memory-bandwidth bar chart for every results/*.json file.

Usage:  python tools/plot.py          (needs matplotlib)
Writes docs/bandwidth_<gpu>.png. Kernels without a bandwidth figure (matmul) are skipped.
"""
import glob
import json
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OURS, REFERENCE, PEAK = "#1a73e8", "#9aa0a6", "#d93025"


def plot(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [r for r in data["results"] if r.get("bandwidth_gbs") is not None]
    if not rows:
        return None
    names = [r["kernel"] for r in rows]
    values = [r["bandwidth_gbs"] for r in rows]
    colors = [REFERENCE if n.startswith("torch.") else OURS for n in names]
    peak = data.get("peak_bandwidth_gbs")

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6))
    positions = list(range(len(rows)))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    for y, v in zip(positions, values):
        ax.text(v + 4, y, f"{v:.0f}", va="center", fontsize=9)
    if peak:
        ax.axvline(peak, color=PEAK, linestyle="--", linewidth=1)
        ax.text(peak, -0.7, f"peak {peak:g} GB/s", color=PEAK, ha="right", fontsize=9)
    ax.set_xlim(0, (peak or max(values)) * 1.12)
    ax.set_xlabel("Effective memory bandwidth (GB/s), higher is better")
    ax.set_title(f"cuda-mini kernels on {data['gpu']}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(ROOT, "docs", f"bandwidth_{stem}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        out = plot(path)
        if out:
            print("Wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
