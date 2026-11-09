"""Physics and compatibility tests for the v2 "blocks" world (src/pefe_blocks.py).

Run: python tests/test_blocks.py      (about 30 seconds)
 or: pytest tests/test_blocks.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402

import pefe_np as P  # noqa: E402
import pefe_blocks as BL  # noqa: E402

CFG = BL.BlockEnvConfig()  # default v2 settings
RMIN = CFG.agent_radius + CFG.bar_radius


def _setup(agents, bars, cfg=CFG):
    """One arena.  agents: list of (x, y, heading); bars: list of (x, y, angle).
    Food is removed so nothing else happens."""
    import dataclasses
    cfg = dataclasses.replace(cfg, n_agents=len(agents), n_bars=len(bars), n_food=1)
    env = BL.BlockEnv(1, cfg)
    env.reset([0])
    env.alive[:] = False
    env.pos[0] = np.array([a[:2] for a in agents], np.float32)
    env.theta[0] = np.array([a[2] for a in agents], np.float32)
    env.vel[:] = 0
    env.bc[0] = np.array([b[:2] for b in bars], float)
    env.bphi[0] = np.array([b[2] for b in bars], float)
    env.bv[:] = 0
    env.bw[:] = 0
    env.bc0, env.bphi0 = env.bc.copy(), env.bphi.copy()
    env._init_stats()
    return env


def _push(env, steps, thrust=1.0):
    """Every agent thrusts straight ahead at full power."""
    act = np.zeros((1, env.A, 3))
    act[..., 0] = thrust
    for _ in range(steps):
        env.step(act)
    return env


def _min_gap(env):
    *_, d, _, _ = env._agent_bar_geometry()
    return float(d.min()) if d.size else np.inf


UP = np.pi / 2


# ---------------------------------------------------------------------------
def test_v1_unchanged_with_zero_bars():
    """v2 with 0 bars gives exactly the v1 result (same draws, same physics)."""
    seeds = [1_000_000, 1_000_001, 1_000_002]
    v1 = P.run_batch(None, None, None, P.EnvConfig(), seeds, np.random.default_rng(0),
                     controller=P.handcoded_actions)
    v2 = P.run_batch(None, None, None, BL.BlockEnvConfig(n_bars=0), seeds,
                     np.random.default_rng(0), controller=P.handcoded_actions)
    assert np.array_equal(v1, v2)


def test_food_draws_identical_to_v1():
    seeds = [3, 1_000_000]
    e1, e2 = P.BatchEnv(2, P.EnvConfig()), BL.BlockEnv(2, CFG)
    e1.reset(seeds)
    o2 = e2.reset(seeds)
    assert np.array_equal(e1.food, e2.food)
    assert o2.shape[-1] == 24
    # agents: same draws, except the few nudged out of a bar (moved < RMIN)
    moved = np.sqrt(((e1.pos - e2.pos) ** 2).sum(-1))
    assert moved.max() <= RMIN + 1e-3
    assert np.array_equal(e1.theta, e2.theta)
    assert _min_gap(e2) >= RMIN - 1e-3, "an agent spawned inside a bar"


def test_no_crossing_head_on():
    """An agent driving at full speed into the middle of a bar never ends up
    inside it or on its far side."""
    env = _setup([(500, 450, UP)], [(500, 500, 0.0)])
    side0 = np.sign(env.pos[0, 0, 1] - env.bc[0, 0, 1])
    for _ in range(400):
        _push(env, 1)
        assert _min_gap(env) >= RMIN - 1e-4
        assert np.sign(env.pos[0, 0, 1] - env.bc[0, 0, 1]) == side0


def test_no_crossing_random_crowd():
    """10 agents with random actions among the default bars for 1500 steps:
    no overlaps, bars stay inside the world."""
    env = BL.BlockEnv(2, CFG)
    env.reset([11, 12])
    rng = np.random.default_rng(0)
    W, rb, h = CFG.world_size, CFG.bar_radius, CFG.bar_length / 2
    for t in range(1500):
        act = rng.uniform(-1, 1, (2, CFG.n_agents, 3))
        act[..., 0] = np.abs(act[..., 0])  # mostly forward, so they hit things
        env.step(act)
        assert _min_gap(env) >= RMIN - 1e-3, t
        ends = env.bc[..., None, :] + np.array([-h, h])[:, None] * np.stack(
            [np.cos(env.bphi), np.sin(env.bphi)], -1)[..., None, :]
        assert ends.min() >= rb - 1e-6 and ends.max() <= W - rb + 1e-6


def test_single_push_moves_slightly():
    steps = 300
    env = _push(_setup([(500, 488, UP)], [(500, 500, 0.0)]), steps)
    disp = np.linalg.norm(env.bc[0, 0] - env.bc0[0, 0])
    speed = np.linalg.norm(env.bv[0, 0])
    # free agent would travel ~5 * 300 = 1500; the bar moves ~1-4 % of that
    assert 10 < disp < 0.05 * CFG.max_speed * steps, disp
    assert speed < 0.05 * CFG.max_speed, speed
    assert abs(env.bphi[0, 0]) < 0.02          # centre push: almost no rotation
    assert env.stat_push_agent_steps[0] > 0.9 * steps


def test_end_push_rotates():
    steps = 200
    centre = _push(_setup([(500, 488, UP)], [(500, 500, 0.0)]), steps)
    end = _push(_setup([(580, 488, UP)], [(500, 500, 0.0)]), steps)  # 80 of 100 from centre
    rot_c, rot_e = abs(centre.bphi[0, 0]), abs(end.bphi[0, 0])
    assert rot_e > 0.1 and rot_e > 10 * rot_c, (rot_c, rot_e)
    assert end.bphi[0, 0] > 0  # pushing the right end up turns it counter-clockwise


def test_two_pushers_faster():
    steps = 200
    one = _push(_setup([(500, 488, UP)], [(500, 500, 0.0)]), steps)
    two = _push(_setup([(480, 488, UP), (520, 488, UP)], [(500, 500, 0.0)]), steps)
    d1 = np.linalg.norm(one.bc[0, 0] - one.bc0[0, 0])
    d2 = np.linalg.norm(two.bc[0, 0] - two.bc0[0, 0])
    assert d2 > 1.8 * d1, (d1, d2)
    assert two.stat_multi_bar_steps[0, 0] > 0.9 * steps
    assert two.stat_max_pushers[0, 0] == 2


def test_bar_stops_when_released():
    env = _push(_setup([(500, 488, UP)], [(500, 500, 0.0)]), 100)
    env.pos[0, 0] = (100, 100)
    for _ in range(100):
        env.step(np.zeros((1, 1, 3)))
    assert np.linalg.norm(env.bv[0, 0]) < 1e-3


def test_bar_bar_no_deep_overlap():
    """Push one bar into another: they separate (no passing through)."""
    env = _setup([(470, 488, UP), (500, 488, UP), (530, 488, UP)],
                 [(500, 500, 0.0), (500, 560, 0.0)])
    _push(env, 600)
    gap = env.bc[0, 1, 1] - env.bc[0, 0, 1]
    assert gap > 2 * CFG.bar_radius - 1.0, gap


def test_bar_sensor():
    env = _setup([(500, 450, UP)], [(500, 500, 0.0)])
    obs = env.observe()[0, 0]
    bars = obs[16:24]
    # bar straight ahead (heading up): surface 50 - 5 = 45 away, in sectors
    # 0 and 7 (just left / right of the heading).  Ahead is angle 0 relative.
    assert abs(max(bars[0], bars[7]) - (1 - 45 / CFG.sensor_range)) < 1e-4
    assert bars[3:5].max() == 0  # nothing behind
    assert (bars > 0).sum() >= 2  # a long bar spans several sectors


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK:", name)
