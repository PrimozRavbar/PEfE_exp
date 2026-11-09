"""Render an episode of a clone team (or a baseline) to MP4, plus a trail PNG.

    python src/render.py --genome runs/x/best.npz --out results/best.mp4
    python src/render.py --baseline handcoded --out results/handcoded.mp4
    python src/render.py --genome best.npz --blind --out results/blind.mp4
    python src/render.py --baseline handcoded --world v2 --out results/v2_hand.mp4

The episode uses the first held-out evaluation seed by default.  A 24-input
(v2) genome is rendered in the v2 world with bars automatically: bars are
drawn as brown capsules, their start positions as faint outlines, and a bar
turns orange while two or more agents push it at once.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

import pefe_np as P  # noqa: E402
from evaluate import EVAL_SEEDS, NOISE_SEED, world_config  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


def record_episode(genome=None, cfg=None, baseline=None, seed=EVAL_SEEDS[0], blind=False,
                   steps=1000, ecfg=None):
    ecfg = ecfg or P.EnvConfig(max_steps=steps)
    rec = []
    rng = np.random.default_rng(NOISE_SEED)
    if baseline:
        fn = {"handcoded": P.handcoded_actions, "handcoded_stop": P.handcoded_stop_actions}[baseline]
        P.run_batch(None, None, None, ecfg, [seed], rng, controller=fn, blind=blind, record=rec)
    else:
        gi = np.zeros((1, ecfg.n_agents), int)
        P.run_batch([genome], gi, cfg, ecfg, [seed], rng, blind=blind, record=rec)
    return rec, ecfg


def capsule(x, y, phi, half, r, n=8):
    """Polygon outline of a capsule (segment of half-length `half`, radius r)."""
    a = np.linspace(-np.pi / 2, np.pi / 2, n)
    right = np.stack([half + r * np.cos(a), r * np.sin(a)], -1)
    left = np.stack([-half - r * np.cos(a), -r * np.sin(a)], -1)
    pts = np.concatenate([right, left])
    c, s = np.cos(phi), np.sin(phi)
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def _bar_patches(ax, bars, ecfg, **kw):
    out = []
    for x, y, phi in bars:
        p = Polygon(capsule(x, y, phi, ecfg.bar_length / 2, ecfg.bar_radius), closed=True, **kw)
        ax.add_patch(p)
        out.append(p)
    return out


def render(rec, ecfg, out, title="", every=2, fps=30, trail=60):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set(xlim=(0, ecfg.world_size), ylim=(0, ecfg.world_size), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    has_bars = "bars" in rec[0]
    if has_bars:
        _bar_patches(ax, rec[0]["bars"], ecfg, fill=False, ec="0.8", lw=0.6, ls="--")
        bars = _bar_patches(ax, rec[0]["bars"], ecfg, fc="saddlebrown", ec="none", alpha=0.85,
                            zorder=2)
    food = ax.scatter([], [], s=14, c="green")
    agents = ax.scatter([], [], s=40, c="dodgerblue", zorder=3)
    heads = [ax.plot([], [], color="navy", lw=1.2, zorder=4)[0] for _ in range(ecfg.n_agents)]
    trails = [ax.plot([], [], color="dodgerblue", lw=0.7, alpha=0.5)[0] for _ in range(ecfg.n_agents)]
    sensor = [plt.Circle((0, 0), ecfg.sensor_range, fill=False, ls=":", lw=0.4, color="gray")
              for _ in range(ecfg.n_agents)]
    for c in sensor:
        ax.add_patch(c)
    txt = ax.set_title("")
    pos_hist = np.array([r["pos"] for r in rec])
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    with writer.saving(fig, out, dpi=100):
        for t in range(0, len(rec), every):
            r = rec[t]
            food.set_offsets(r["food"] if len(r["food"]) else np.zeros((0, 2)))
            agents.set_offsets(r["pos"])
            for i in range(ecfg.n_agents):
                p, th = r["pos"][i], r["theta"][i]
                heads[i].set_data([p[0], p[0] + 15 * np.cos(th)], [p[1], p[1] + 15 * np.sin(th)])
                h = pos_hist[max(0, t - trail):t + 1, i]
                trails[i].set_data(h[:, 0], h[:, 1])
                sensor[i].center = p
            if has_bars:
                for p, (x, y, phi), m in zip(bars, r["bars"], r["multi"]):
                    p.set_xy(capsule(x, y, phi, ecfg.bar_length / 2, ecfg.bar_radius))
                    p.set_facecolor("darkorange" if m else "saddlebrown")
            txt.set_text(f"{title}  step {r['step']}  food/agent {r['energy'].mean():.1f}")
            writer.grab_frame()
    plt.close(fig)

    # static summary: full trajectories
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set(xlim=(0, ecfg.world_size), ylim=(0, ecfg.world_size), aspect="equal")
    ax.scatter(rec[0]["food"][:, 0], rec[0]["food"][:, 1], s=8, c="green", alpha=0.5)
    if has_bars:
        _bar_patches(ax, rec[0]["bars"], ecfg, fc="0.85", ec="none")
        _bar_patches(ax, rec[-1]["bars"], ecfg, fc="saddlebrown", ec="none", alpha=0.7)
    for i in range(ecfg.n_agents):
        ax.plot(pos_hist[:, i, 0], pos_hist[:, i, 1], lw=0.8)
        ax.plot(*pos_hist[0, i], "ko", ms=3)
    ax.set_title(f"{title}: trajectories, final food/agent {rec[-1]['energy'].mean():.1f}")
    png = os.path.splitext(out)[0] + "_trails.png"
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome")
    ap.add_argument("--baseline")
    ap.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--every", type=int, default=2, help="record every k-th step")
    ap.add_argument("--title", default=None)
    ap.add_argument("--world", choices=["auto", "v1", "v2"], default="auto",
                    help="auto: v2 for 24-input genomes, else v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.genome:
        g, cfg, meta = P.load_genome(a.genome)
        ecfg = world_config(a.world, g, meta)
        ecfg.max_steps = a.steps
        rec, ecfg = record_episode(g, cfg, seed=a.seed, blind=a.blind, steps=a.steps, ecfg=ecfg)
        title = a.title or os.path.basename(os.path.dirname(a.genome))
    else:
        ecfg = world_config("v1" if a.world == "auto" else a.world)
        ecfg.max_steps = a.steps
        rec, ecfg = record_episode(baseline=a.baseline, seed=a.seed, blind=a.blind,
                                   steps=a.steps, ecfg=ecfg)
        title = a.title or a.baseline
    if a.blind:
        title += " (blind)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    png = render(rec, ecfg, a.out, title, every=a.every)
    print("wrote", a.out, png, "final food/agent", rec[-1]["energy"].mean())


if __name__ == "__main__":
    main()
