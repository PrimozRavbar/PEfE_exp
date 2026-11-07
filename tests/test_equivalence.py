"""BatchEnv must reproduce environment3.ForagingEnv exactly.

Run: python tests/test_equivalence.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from agent3 import Agent
from environment3 import ForagingEnv
import pefe_np as P


def run_ref(seed, policy, steps):
    agents = [Agent() for _ in range(10)]
    env = ForagingEnv(agents, n_food=200, max_steps=steps)
    obs, _ = env.reset(seed=seed)
    traj = []
    for _ in range(steps):
        act = policy(obs[None])[0]
        obs, r, term, trunc, _ = env.step(act)
        traj.append(obs.copy())
        if term or trunc:
            break
    return np.array([a.energy for a in agents]), np.array(traj)


def run_new(seed, policy, steps):
    env = P.BatchEnv(1, P.EnvConfig(max_steps=steps))
    obs = env.reset([seed])
    traj = []
    for _ in range(steps):
        obs, _ = env.step(policy(obs))
        traj.append(obs[0].copy())
    return env.energy[0], np.array(traj)


def wobbly(obs):
    a = P.handcoded_actions(obs)
    a[..., 2] = 0.5  # exercise turning too
    return a


if __name__ == "__main__":
    for pol in (P.handcoded_actions, wobbly):
        for seed in (1, 2, 3):
            e1, t1 = run_ref(seed, pol, 400)
            e2, t2 = run_new(seed, pol, 400)
            n = min(len(t1), len(t2))
            assert np.allclose(e1, e2), (pol.__name__, seed, e1, e2)
            err = np.abs(t1[:n] - t2[:n]).max()
            assert err < 1e-4, err
            print(pol.__name__, seed, "food", e1.sum(), "obs max err", err)
    print("OK: BatchEnv matches environment3")


def brain3_reference_step(g, h, x, cfg):
    """Line-by-line transcription of Brain3.forward (noise off)."""
    hp = g.hyper
    hm = hp[0] * cfg.hidden_max
    sensory = np.clip(g.Win @ x, 0, hm)
    recurrent = g.Wh @ h
    h = h - hp[4] * cfg.leak_hidden * h
    h = np.clip(h, 0, hm)
    h = h + hp[2] * cfg.add_sens * sensory + hp[1] * cfg.add_hidden * recurrent
    h = np.clip(h, 0, hm)
    return h, np.tanh(g.Wout @ h)


def test_brain():
    rng = np.random.default_rng(0)
    cfg = P.BrainConfig(noise=False)
    gs = []
    for k in range(3):
        g = P.Genome.zeros()
        for _ in range(40):
            g.add_remove_connection(rng, 0.01, 0.5, caps=False)
        g.hyper[:6] = rng.uniform(0.5, 1.5, 6)
        gs.append(g)
    gs.append(P.Genome.random_dense(rng))  # dense path
    idx = [0, 1, 2, 3, 0]
    bb = P.BatchBrain(gs, idx, cfg, rng)
    hs = [np.zeros(128) for _ in idx]
    for t in range(50):
        x = rng.random((len(idx), 16))
        out = bb(x)
        for n, gi in enumerate(idx):
            hs[n], o = brain3_reference_step(gs[gi], hs[n], x[n], cfg)
            # float32 (reference) vs float64 (batched) rounding only; the
            # recurrence can be chaotic, so compare with a relative norm.
            rel = np.linalg.norm(hs[n] - bb.h[n]) / (np.linalg.norm(hs[n]) + 1e-9)
            assert rel < 1e-4, (t, n, rel)
            assert np.abs(o - out[n]).max() < 1e-3, (t, n)
    print("OK: BatchBrain matches Brain3 equations")


test_brain()
