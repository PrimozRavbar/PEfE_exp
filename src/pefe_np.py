"""Numpy port of the "3" stack, batched across arenas and brains.

Why this file exists
--------------------
* torch cannot be installed in the development container, and the code only
  uses a handful of tensor ops, so Brain3 / GenomePE(full_rank=True) are ported
  to numpy here (same equations, same default constants).
* The original runner steps every brain separately in a Python loop and every
  arena separately.  Here `BatchEnv` steps E arenas x A agents at once and
  `BatchBrain` steps all E*A brains at once with a sparse gather/bincount
  matvec, which is ~50x faster for sparse genomes.

`BatchEnv` reproduces environment3.ForagingEnv exactly (same reset RNG draw
order, same integration, same sensor sectors and food-assignment rule); see
tests/test_equivalence.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

INPUT_SIZE = 16
HIDDEN_SIZE = 128
OUTPUT_SIZE = 3
NUM_HYPERPARAMETERS = 20
N_WIN = HIDDEN_SIZE * INPUT_SIZE
N_WH = HIDDEN_SIZE * HIDDEN_SIZE
N_WOUT = OUTPUT_SIZE * HIDDEN_SIZE
NUM_PARAMETERS_FULL = N_WIN + N_WH + N_WOUT
# genome_pe3 caps: 20% of each matrix
MAX_WIN = N_WIN // 5
MAX_WH = N_WH // 5
MAX_WOUT = N_WOUT // 5


# ----------------------------------------------------------------------------
# Brain configuration (the constants Brain3 hard-codes)
# ----------------------------------------------------------------------------
@dataclass
class BrainConfig:
    """Scale factors multiplying genome hyperparameters 0..5 (Brain3 defaults).

    effective value = hyper[i] * scale, exactly like Brain3.
    """

    hidden_max: float = 10000.0
    add_hidden: float = 0.5
    add_sens: float = 0.01
    leak_sens: float = 0.1  # unused by Brain3's forward pass; kept for parity
    leak_hidden: float = 0.1
    max_epsilon: float = 1.0
    eps_unit: int = 79       # hidden unit that drives output noise
    eps_gain: float = 100.0
    noise: bool = True       # Brain3 output noise on/off

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s)) if s else cls()


# ----------------------------------------------------------------------------
# Genome: sparse full-rank weights + 20 hyperparameters
# ----------------------------------------------------------------------------
class Genome:
    """Full-rank genome, numpy port of GenomePE(full_rank=True).

    Zeros mean "no connection".  Differences from genome_pe3 (bugs fixed):
      * clone() copies arrays instead of building a random genome first;
      * hyperparameters are clamped to stay > 0 (a hidden_max <= 0 after a
        mutation silently killed the brain).
    """

    __slots__ = ("Win", "Wh", "Wout", "hyper")

    def __init__(self, Win=None, Wh=None, Wout=None, hyper=None, n_inputs=INPUT_SIZE):
        # n_inputs: 16 for the v1 world; the v2 "blocks" world (pefe_blocks.py)
        # adds 8 bar sensors, so its genomes are built with n_inputs=24.
        self.Win = np.zeros((HIDDEN_SIZE, n_inputs), np.float32) if Win is None else Win
        self.Wh = np.zeros((HIDDEN_SIZE, HIDDEN_SIZE), np.float32) if Wh is None else Wh
        self.Wout = np.zeros((OUTPUT_SIZE, HIDDEN_SIZE), np.float32) if Wout is None else Wout
        self.hyper = np.ones(NUM_HYPERPARAMETERS, np.float32) if hyper is None else hyper

    # -- construction --------------------------------------------------------
    @classmethod
    def zeros(cls, n_inputs=INPUT_SIZE):
        return cls(n_inputs=n_inputs)

    @classmethod
    def random_dense(cls, rng, scale=0.1, n_inputs=INPUT_SIZE):
        """Same as GenomePE(full_rank=True, initialization='random')."""
        return cls(
            (rng.standard_normal((HIDDEN_SIZE, n_inputs)) * scale).astype(np.float32),
            (rng.standard_normal((HIDDEN_SIZE, HIDDEN_SIZE)) * scale).astype(np.float32),
            (rng.standard_normal((OUTPUT_SIZE, HIDDEN_SIZE)) * scale).astype(np.float32),
        )

    def clone(self):
        return Genome(self.Win.copy(), self.Wh.copy(), self.Wout.copy(), self.hyper.copy())

    # -- stats ---------------------------------------------------------------
    def n_connections(self):
        return int(np.count_nonzero(self.Win) + np.count_nonzero(self.Wh)
                   + np.count_nonzero(self.Wout))

    def connection_counts(self):
        return {"Win": int(np.count_nonzero(self.Win)),
                "Wh": int(np.count_nonzero(self.Wh)),
                "Wout": int(np.count_nonzero(self.Wout))}

    @property
    def n_inputs(self):
        return self.Win.shape[1]

    def matrices(self):
        # caps: 20% of each matrix (== MAX_WIN/MAX_WH/MAX_WOUT for 16 inputs)
        return (("Win", self.Win, self.Win.size // 5), ("Wh", self.Wh, MAX_WH),
                ("Wout", self.Wout, MAX_WOUT))

    # -- mutation (faithful to genome_pe3; extra knobs default to old behavior)
    def mutate(self, rng, mutation_rate=0.1, mutation_scale=0.05,
               mutate_hyper=True):
        """Perturb existing (nonzero) weights; perturb hyperparameters."""
        for _, W, _ in self.matrices():
            mask = (W != 0) & (rng.random(W.shape) < mutation_rate)
            if mask.any():
                W[mask] += (rng.standard_normal(int(mask.sum())) * mutation_scale).astype(np.float32)
        if mutate_hyper:
            hm = rng.random(NUM_HYPERPARAMETERS) < mutation_rate
            self.hyper[hm] += (rng.standard_normal(int(hm.sum())) * mutation_scale).astype(np.float32)
            np.maximum(self.hyper, 1e-3, out=self.hyper)

    def add_remove_connection(self, rng, mutation_rate=0.002, mutation_scale=1.0,
                              remove_rate=None, caps=True, add_per_child=None,
                              add_share=None):
        """Structural mutation.

        Default (remove_rate=None) is genome_pe3's rule: every entry is hit
        with probability `mutation_rate`; hit nonzeros are removed, hit zeros
        are added (up to the 20 % cap).  `remove_rate` lets the remove
        probability differ from the add probability.

        `add_per_child` (optional) replaces the per-entry add probability by
        a fixed *expected number* of additions per call, split over
        (Win, Wh, Wout) by `add_share` (default: proportional to matrix size,
        i.e. uniform over entries).  Within a matrix each empty entry is added
        with probability expected_adds_m / n_empty_m, so the add count does
        not depend on how full the genome already is.
        """
        rr = mutation_rate if remove_rate is None else remove_rate
        if add_per_child is not None:
            share = (np.array([self.Win.size, N_WH, N_WOUT], float) if add_share is None
                     else np.asarray(add_share, float))
            share = share / share.sum()
        for m, (_, W, cap) in enumerate(self.matrices()):
            nz = W != 0
            remove = nz & (rng.random(W.shape) < rr)
            if add_per_child is None:
                p_add = mutation_rate
            else:
                n_empty = W.size - int(nz.sum())
                p_add = min(1.0, add_per_child * share[m] / max(n_empty, 1))
            add = (~nz) & (rng.random(W.shape) < p_add)
            W[remove] = 0.0
            add_idx = np.flatnonzero(add)
            if caps:
                available = cap - int(np.count_nonzero(W))
                if available <= 0:
                    continue
                if len(add_idx) > available:
                    add_idx = rng.permutation(add_idx)[:available]
            if len(add_idx):
                W.flat[add_idx] = (rng.standard_normal(len(add_idx)) * mutation_scale).astype(np.float32)
                # an exact zero would be "no connection"; vanishingly rare
                W.flat[add_idx[W.flat[add_idx] == 0]] = 1e-6

    def add_path(self, rng, mutation_scale=1.0):
        """Add one input->hidden->output path through a random hidden unit:
        a new Win[j, i] and a new Wout[k, j] (entries that already exist are
        left alone).  Off unless evolve.py --add-path-prob > 0."""
        j = int(rng.integers(HIDDEN_SIZE))
        i = int(rng.integers(self.n_inputs))
        k = int(rng.integers(OUTPUT_SIZE))
        for W, idx in ((self.Win, (j, i)), (self.Wout, (k, j))):
            if W[idx] == 0:
                W[idx] = np.float32(rng.standard_normal() * mutation_scale) or np.float32(1e-6)

    # -- io ------------------------------------------------------------------
    def to_dict(self, prefix=""):
        return {prefix + "Win": self.Win, prefix + "Wh": self.Wh,
                prefix + "Wout": self.Wout, prefix + "hyper": self.hyper}

    @classmethod
    def from_dict(cls, d, prefix=""):
        return cls(np.array(d[prefix + "Win"], np.float32), np.array(d[prefix + "Wh"], np.float32),
                   np.array(d[prefix + "Wout"], np.float32), np.array(d[prefix + "hyper"], np.float32))


def save_genome(path, genome, brain_cfg: BrainConfig, meta=None):
    np.savez_compressed(path, **genome.to_dict(), brain_config=brain_cfg.to_json(),
                        meta=json.dumps(meta or {}))


def load_genome(path):
    d = np.load(path, allow_pickle=False)
    g = Genome.from_dict(d)
    cfg = BrainConfig.from_json(str(d["brain_config"])) if "brain_config" in d else BrainConfig()
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return g, cfg, meta


# ----------------------------------------------------------------------------
# Batched brain
# ----------------------------------------------------------------------------
class _SparseBlock:
    """Block-diagonal sparse matrix for N brains: y[n] = W_{g(n)} x[n]."""

    def __init__(self, mats, genome_of_agent, n_rows, n_cols):
        rows, cols, vals = [], [], []
        for n, g in enumerate(genome_of_agent):
            r, c = np.nonzero(mats[g])
            rows.append(r + n * n_rows)
            cols.append(c + n * n_cols)
            vals.append(mats[g][r, c])
        self.rows = np.concatenate(rows) if rows else np.zeros(0, int)
        self.cols = np.concatenate(cols) if cols else np.zeros(0, int)
        self.vals = (np.concatenate(vals) if vals else np.zeros(0)).astype(np.float64)
        self.N, self.n_rows = len(genome_of_agent), n_rows

    def __call__(self, x):  # x: (N, n_cols)
        flat = x.reshape(-1)
        y = np.bincount(self.rows, weights=self.vals * flat[self.cols],
                        minlength=self.N * self.n_rows)
        return y.reshape(self.N, self.n_rows)


class _DenseBlock:
    def __init__(self, mats, genome_of_agent):
        self.W = np.stack([mats[g] for g in genome_of_agent]).astype(np.float64)

    def __call__(self, x):
        return np.einsum("nij,nj->ni", self.W, x)


def _block(mats, gidx, n_rows, n_cols):
    nnz = sum(np.count_nonzero(mats[g]) for g in gidx)
    if nnz > 0.15 * len(gidx) * n_rows * n_cols:
        return _DenseBlock(mats, gidx)
    return _SparseBlock(mats, gidx, n_rows, n_cols)


class BatchBrain:
    """Brain3 forward pass for N agents at once (numpy port).

    `genome_of_agent[n]` is the index into `genomes` driving agent n.  Each
    genome may carry its own BrainConfig (`cfgs`), but by default one config
    is shared.
    """

    def __init__(self, genomes, genome_of_agent, cfg: BrainConfig, rng):
        self.cfg = cfg
        self.rng = rng
        gidx = list(genome_of_agent)
        self.N = len(gidx)
        self.n_inputs = genomes[gidx[0]].n_inputs
        self.Win = _block([g.Win for g in genomes], gidx, HIDDEN_SIZE, self.n_inputs)
        self.Wh = _block([g.Wh for g in genomes], gidx, HIDDEN_SIZE, HIDDEN_SIZE)
        self.Wout = _block([g.Wout for g in genomes], gidx, OUTPUT_SIZE, HIDDEN_SIZE)
        hp = np.stack([genomes[g].hyper[:6] for g in gidx]).astype(np.float64)
        self.hidden_max = (hp[:, 0] * cfg.hidden_max)[:, None]
        self.add_hidden = (hp[:, 1] * cfg.add_hidden)[:, None]
        self.add_sens = (hp[:, 2] * cfg.add_sens)[:, None]
        self.leak_hidden = (hp[:, 4] * cfg.leak_hidden)[:, None]
        self.max_eps = hp[:, 5] * cfg.max_epsilon
        self.h = np.zeros((self.N, HIDDEN_SIZE))

    def reset(self):
        self.h[:] = 0

    def __call__(self, obs):
        hm = self.hidden_max
        sensory = np.clip(self.Win(obs), 0, hm)
        recurrent = self.Wh(self.h)
        h = self.h - self.leak_hidden * self.h
        h = np.clip(h, 0, hm)
        h = h + self.add_sens * sensory + self.add_hidden * recurrent
        self.h = np.clip(h, 0, hm)
        out = self.Wout(self.h)
        if self.cfg.noise:
            eps = np.clip(self.h[:, self.cfg.eps_unit] * self.cfg.eps_gain, 0, self.max_eps)
            if np.any(eps > 0):
                out = out + self.rng.standard_normal(out.shape) * out * eps[:, None]
        return np.tanh(out)


# ----------------------------------------------------------------------------
# Batched environment (exact replica of environment3.ForagingEnv)
# ----------------------------------------------------------------------------
@dataclass
class EnvConfig:
    n_agents: int = 10
    n_food: int = 200
    max_steps: int = 1000
    world_size: float = 1000.0
    food_radius: float = 8.0
    food_energy: float = 1.0
    sensor_range: float = 150.0
    max_speed: float = 5.0
    acceleration: float = 1.0
    friction: float = 0.95
    turn_rate: float = 0.15
    agent_radius: float = 5.0

    n_inputs = INPUT_SIZE  # sensor count (class attribute, not a field)

    def make_env(self, n_arenas):
        return BatchEnv(n_arenas, self)


class BatchEnv:
    def __init__(self, n_arenas, cfg: EnvConfig):
        self.E, self.A, self.cfg = n_arenas, cfg.n_agents, cfg

    def reset(self, seeds):
        c, E, A = self.cfg, self.E, self.A
        self.pos = np.zeros((E, A, 2), np.float32)
        self.vel = np.zeros((E, A, 2), np.float32)
        self.theta = np.zeros((E, A), np.float32)
        self.food = np.zeros((E, c.n_food, 2), np.float32)
        margin = c.agent_radius
        for e, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            for a in range(A):  # same draw order as environment3.reset
                self.pos[e, a] = rng.uniform(margin, c.world_size - margin, size=2)
                self.theta[e, a] = rng.uniform(-np.pi, np.pi)
            self.food[e] = rng.uniform(c.food_radius, c.world_size - c.food_radius,
                                       size=(c.n_food, 2)).astype(np.float32)
        self.alive = np.ones((E, c.n_food), bool)
        self.energy = np.zeros((E, A), np.float64)
        self.steps = 0
        return self.observe()

    def _sector_max(self, dx, dy, d2, extra_mask=None):
        R = self.cfg.sensor_range
        vis = (d2 > 1e-12) & (d2 <= R * R)
        if extra_mask is not None:
            vis &= extra_mask
        E, A = self.pos.shape[:2]
        e, a, k = np.nonzero(vis)  # only visible pairs (~5% of all)
        ddx, ddy = dx[e, a, k], dy[e, a, k]
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        ang = (np.arctan2(ddy, ddx) - self.theta[e, a]) % (2 * np.pi)
        sec = (ang * (8 / (2 * np.pi))).astype(np.int64) % 8
        strength = 1.0 - dist / R
        tmp = np.zeros(E * A * 8, np.float32)  # obs starts at zero
        np.maximum.at(tmp, (e * A + a) * 8 + sec, strength)
        return tmp.reshape(E, A, 8)

    def _food_delta(self):
        dx = self.food[:, None, :, 0] - self.pos[:, :, None, 0]  # (E,A,F)
        dy = self.food[:, None, :, 1] - self.pos[:, :, None, 1]
        return dx, dy, dx * dx + dy * dy

    def observe(self, food_delta=None):
        dx, dy, d2 = self._food_delta() if food_delta is None else food_delta
        obs_food = self._sector_max(dx, dy, d2, self.alive[:, None, :])
        ax = self.pos[:, None, :, 0] - self.pos[:, :, None, 0]
        ay = self.pos[:, None, :, 1] - self.pos[:, :, None, 1]
        obs_ag = self._sector_max(ax, ay, ax * ax + ay * ay)
        return np.concatenate([obs_food, obs_ag], -1)

    def step(self, actions):
        self._move_agents(actions)
        return self._eat()

    def _move_agents(self, actions):
        c = self.cfg
        act = np.clip(np.asarray(actions, np.float32), -1, 1)
        theta = self.theta + act[..., 2] * c.turn_rate
        ct, st = np.cos(theta), np.sin(theta)
        ax = act[..., 0] * ct - act[..., 1] * st
        ay = act[..., 0] * st + act[..., 1] * ct
        vel = c.friction * self.vel + c.acceleration * np.stack([ax, ay], -1)
        speed = np.sqrt((vel * vel).sum(-1))
        fast = speed > c.max_speed
        if fast.any():
            vel[fast] *= (c.max_speed / speed[fast])[:, None]
        pos = self.pos + vel
        lo, hi = c.agent_radius, c.world_size - c.agent_radius
        below, above = pos < lo, pos > hi
        pos = np.clip(pos, lo, hi)
        vel[below & (vel < 0)] = 0
        vel[above & (vel > 0)] = 0
        self.theta, self.pos, self.vel = theta.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32)

    def _eat(self):
        c = self.cfg
        # food: each food goes to the closest colliding agent
        fd = self._food_delta()
        d2 = fd[2]  # (E,A,F), reused by observe()
        coll = (d2 <= (c.agent_radius + c.food_radius) ** 2) & self.alive[:, None, :]
        rewards = np.zeros(self.pos.shape[:2], np.float64)
        if coll.any():
            consumed = coll.any(1)  # (E,F)
            winners = np.argmin(np.where(coll, d2, np.inf), axis=1)  # (E,F)
            e_idx, f_idx = np.nonzero(consumed)
            np.add.at(rewards, (e_idx, winners[e_idx, f_idx]), c.food_energy)
            self.alive[consumed] = False
        self.energy += rewards
        self.steps += 1
        return self.observe(fd), rewards


# ----------------------------------------------------------------------------
# Controllers and episode runner
# ----------------------------------------------------------------------------
SECTOR_CENTERS = (np.arange(8) + 0.5) * (2 * np.pi / 8)


def handcoded_actions(obs):
    """The brief's baseline, literally: thrust toward the strongest food sensor,
    action = [cos a, sin a, 0].  With nothing in view argmax picks sector 0, so
    the agent keeps thrusting (drifts) -- that is what gives 19.8."""
    k = np.argmax(obs[..., :8], -1)
    a = SECTOR_CENTERS[k]
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)


def handcoded_stop_actions(obs):
    """Same, but stand still when no food is in view (no exploration)."""
    act = handcoded_actions(obs)
    act[obs[..., :8].max(-1) <= 0] = 0.0
    return act


def run_batch(genomes, genome_of_agent, brain_cfg, env_cfg, seeds, rng,
              controller=None, blind=False, record=None, env_out=None,
              zero_inputs=None):
    """Run len(seeds) arenas in parallel.

    genome_of_agent: int array (E, A) of indices into `genomes`.
    controller: None -> brains; or a function obs(E,A,n_in) -> actions(E,A,3).
    blind: zero the food sensors (luck / drift check).
    record: optional list; per-step dict of positions etc. for arena 0.
    zero_inputs: optional index/slice of inputs to zero every step (e.g. the
      v2 bar sensors, slice(16, 24)).
    env_out: optional list; the finished environment is appended (v2 uses it
      to read bar statistics).
    env_cfg.make_env picks the world: EnvConfig -> v1 BatchEnv,
      pefe_blocks.BlockEnvConfig -> v2 BlockEnv.
    Returns food collected per agent, shape (E, A).
    """
    E = len(seeds)
    env = env_cfg.make_env(E)
    n_in = env_cfg.n_inputs
    obs = env.reset(seeds)
    brain = None
    if controller is None:
        gidx = np.asarray(genome_of_agent).reshape(-1)
        brain = BatchBrain(genomes, gidx, brain_cfg, rng)
    A = env_cfg.n_agents
    for t in range(env_cfg.max_steps):
        if blind:
            obs = obs.copy()
            obs[..., :8] = 0
        if zero_inputs is not None:
            obs = obs.copy()
            obs[..., zero_inputs] = 0
        if record is not None:
            record.append(_snapshot(env, t))
        if brain is None:
            act = controller(obs)
        else:
            act = brain(obs.reshape(E * A, n_in).astype(np.float64)).reshape(E, A, 3)
        obs, _ = env.step(act)
        if not env.alive.any():
            break
    if record is not None:
        record.append(_snapshot(env, env.steps))
    if env_out is not None:
        env_out.append(env)
    return env.energy.copy()


def _snapshot(env, t):
    r = {"pos": env.pos[0].copy(), "theta": env.theta[0].copy(),
         "food": env.food[0][env.alive[0]].copy(),
         "energy": env.energy[0].copy(), "step": t}
    if hasattr(env, "snapshot_extra"):
        r.update(env.snapshot_extra())
    return r
