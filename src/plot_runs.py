"""Plot fitness and connection curves for groups of runs.

Runs are grouped by --group LABEL=GLOB (repeatable); each group is drawn with
its seeds as thin lines and the across-seed mean as a thick line.

    python src/plot_runs.py --group default='runs/ref_default_s*' \
        --group gainfix='runs/A_gain*_s*' --out results/fitness.png
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except ValueError:
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", default="generation", choices=["generation", "evals"])
    ap.add_argument("--title", default="")
    ap.add_argument("--handcoded", type=float, default=19.70)
    ap.add_argument("--random", type=float, default=7.71)
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for gi, spec in enumerate(a.group):
        label, pattern = spec.split("=", 1)
        logs = [read_log(os.path.join(d, "log.csv")) for d in sorted(glob.glob(pattern))
                if os.path.exists(os.path.join(d, "log.csv"))]
        if not logs:
            print("no runs for", spec)
            continue
        c = colors[gi % len(colors)]
        for key, ax in (("mean", axes[0]), ("valid", axes[1]), ("mean_connections", axes[2])):
            ys = []
            for L in logs:
                x, y = L[a.x], L[key]
                m = ~np.isnan(y)
                ax.plot(x[m], y[m], color=c, alpha=0.3, lw=0.8)
                ys.append((x[m], y[m]))
            n = min(len(y) for _, y in ys)
            ax.plot(ys[0][0][:n], np.mean([y[:n] for _, y in ys], 0), color=c, lw=2.2,
                    label=f"{label} (n={len(logs)})")
    axes[0].set_title("population mean food / agent (mixed arenas)")
    axes[1].set_title("best genome, clone-team validation score")
    axes[2].set_title("mean connections per genome")
    for ax in axes[:2]:
        ax.axhline(a.handcoded, color="k", ls="--", lw=1)
        ax.axhline(a.random, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 21)
    axes[1].text(0.01, a.handcoded + 0.2, "hand-coded", transform=axes[1].get_yaxis_transform(), fontsize=8)
    axes[1].text(0.01, a.random + 0.2, "random dense net", transform=axes[1].get_yaxis_transform(), fontsize=8)
    for ax in axes:
        ax.set_xlabel(a.x)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
