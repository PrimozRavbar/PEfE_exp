"""v2 world: v1 foraging plus heavy movable bars ("blocks").

Everything here is opt-in.  The v1 world (pefe_np.EnvConfig / BatchEnv) is
untouched; `BlockEnvConfig` is a subclass of `EnvConfig` whose `make_env`
returns a `BlockEnv`, so `pefe_np.run_batch` and the evolution / evaluation
scripts run either world from the config they are given.

Bars
----
* A bar is a capsule: a segment of length `bar_length` with radius
  `bar_radius` (so it is 2*bar_radius thick).  State per bar: centre, angle,
  linear velocity, angular velocity.  Mass `bar_mass` (an agent has mass 1),
  moment of inertia of a thin rod, M L^2 / 12.  Every step the bar's linear and
  angular velocities are multiplied by `bar_damping` / `bar_ang_damping`
  (strong damping: a bar stops within a few dozen steps when nobody pushes).
* Agent-bar contact (circle vs capsule): if an agent overlaps a bar, an
  inelastic impulse along the contact normal cancels the approaching part of
  the relative velocity, using the full rigid-body effective mass
  1 / (1/m + 1/M + (r x n)^2 / I).  The bar receives the opposite impulse, so
  pushing off-centre gives a torque.  Impulses of all agents touching a bar in
  the same step are summed, so several pushers add up.  The agent is then
  moved out of the bar completely (the bar is not moved by this position
  correction, so position correction adds no momentum).  If an agent is
  squeezed between a bar and a wall, the remaining overlap is removed by
  moving the bar.
* No tunnelling: an agent moves at most `max_speed` = 5 per step, which is
  less than agent_radius + bar_radius = 10, so it can never jump past a
  bar's axis within one step and is always pushed back to the side it came
  from.
* Bars stay inside the world: after every step the centre is clamped so the
  whole capsule is inside [0, world_size]^2, and the velocity component into
  the wall is zeroed.
* Bar-bar overlap (the simple choice, documented): bars spawn with a gap of
  at least `bar_gap` between their surfaces.  During the episode, each
  overlapping pair (closest distance between their axes < 2*bar_radius,
  measured from the four endpoints, which is exact unless the axes cross --
  impossible at these speeds) gets an inelastic impulse along the contact
  normal and is pushed apart by half the overlap each (translation only).
  One pass per step, so chains of bars can overlap slightly for a step.
* Bars do not block sensing (food and agents are seen through them), and
  agents still pass through each other as in v1.

Sensors (24 inputs): [8 food, 8 agents, 8 bars].  Bar sector k holds
1 - d / sensor_range for the nearest bar *surface* point in that sector
(d = distance to the axis point minus bar_radius), computed over points
spaced <= `bar_sample_spacing` along every bar axis plus the exact nearest
point of every bar.  0 = no bar surface within range in that sector.

Random streams: food and agent positions come from exactly the same
`default_rng(seed)` draws as v1.  Bars come from a separate stream
`default_rng([seed, BAR_STREAM])`.  After both are drawn, any agent that
spawned inside a bar is nudged out along the contact normal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass  # noqa: F401 (asdict re-exported)

import numpy as np

import pefe_np as P

BAR_STREAM = 0xB10C5  # second SeedSequence word for the bar stream
N_INPUTS_V2 = 24


@dataclass
class BlockEnvConfig(P.EnvConfig):
    n_bars: int = 40
    bar_length: float = 300.0
    bar_radius: float = 5.0
    bar_mass: float = 50.0
    bar_damping: float = 0.9       # per-step multiplier on linear velocity
    bar_ang_damping: float = 0.9   # per-step multiplier on angular velocity
    bar_gap: float = 12.0          # min surface gap at spawn (bar-bar, bar-wall)
    bar_sample_spacing: float = 10.0

    n_inputs = N_INPUTS_V2

    def make_env(self, n_arenas):
        return BlockEnv(n_arenas, self)

    @property
    def bar_inertia(self):
        return self.bar_mass * self.bar_length ** 2 / 12.0


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _point_segment(px, py, cx, cy, ux, uy, h):
    """Closest point on the segment c + s u, |s| <= h, to p (broadcasting).
    Returns (s, qx, qy, dx, dy, d) with d = |p - q|."""
    s = np.clip((px - cx) * ux + (py - cy) * uy, -h, h)
    qx, qy = cx + s * ux, cy + s * uy
    dx, dy = px - qx, py - qy
    return s, qx, qy, dx, dy, np.sqrt(dx * dx + dy * dy)


class BlockEnv(P.BatchEnv):
    """v1 BatchEnv plus bars.  Arrays: bc (E,B,2) centres, bphi (E,B) angles,
    bv (E,B,2) velocities, bw (E,B) angular velocities (all float64)."""

    # ------------------------------------------------------------------ reset
    def reset(self, seeds):
        self._spawn_bars(seeds)
        super().reset(seeds)          # identical food / agent draws to v1
        self._nudge_agents_out()
        self._init_stats()
        return self.observe()

    def _spawn_bars(self, seeds):
        c, E, B = self.cfg, self.E, self.cfg.n_bars
        h, rb, W = c.bar_length / 2, c.bar_radius, c.world_size
        self.bc = np.zeros((E, B, 2))
        self.bphi = np.zeros((E, B))
        self.bv = np.zeros((E, B, 2))
        self.bw = np.zeros((E, B))
        self.n_bars_placed = np.zeros(E, int)
        for e, s in enumerate(seeds):
            rng = np.random.default_rng([int(s), BAR_STREAM])
            placed = []
            # rejection sampling, 64 candidates at a time (first valid one wins)
            for _ in range(500):
                if len(placed) == B:
                    break
                phi = rng.uniform(0, np.pi, 64)
                ex = h * np.abs(np.cos(phi)) + rb + c.bar_gap
                ey = h * np.abs(np.sin(phi)) + rb + c.bar_gap
                fx, fy = rng.random(64), rng.random(64)
                cx, cy = ex + fx * (W - 2 * ex), ey + fy * (W - 2 * ey)
                ok = (2 * ex < W) & (2 * ey < W)
                if placed:
                    pl = np.array(placed)
                    dist = _segment_distance(cx[:, None], cy[:, None], phi[:, None],
                                             pl[None, :, 0], pl[None, :, 1], pl[None, :, 2], h)
                    ok &= dist.min(1) >= 2 * rb + c.bar_gap
                if ok.any():
                    i = int(np.argmax(ok))
                    placed.append((cx[i], cy[i], phi[i]))
            if len(placed) < B:
                raise RuntimeError(f"could only place {len(placed)} of {B} bars "
                                   f"(length {c.bar_length}); use fewer or shorter bars")
            for b, (cx, cy, phi) in enumerate(placed):
                self.bc[e, b] = cx, cy
                self.bphi[e, b] = phi
            self.n_bars_placed[e] = len(placed)
        self.bc0 = self.bc.copy()
        self.bphi0 = self.bphi.copy()

    def _agent_bar_geometry(self, pos=None):
        """Agent-bar closest points.  Shapes (E,A,B)."""
        c = self.cfg
        pos = self.pos.astype(np.float64) if pos is None else pos
        ux, uy = np.cos(self.bphi)[:, None], np.sin(self.bphi)[:, None]
        cx, cy = self.bc[:, None, :, 0], self.bc[:, None, :, 1]
        px, py = pos[:, :, None, 0], pos[:, :, None, 1]
        s, qx, qy, dx, dy, d = _point_segment(px, py, cx, cy, ux, uy, c.bar_length / 2)
        return s, qx, qy, dx, dy, d, ux, uy

    def _nudge_agents_out(self):
        c = self.cfg
        if c.n_bars == 0:
            return
        rmin = c.agent_radius + c.bar_radius
        lo, hi = c.agent_radius, c.world_size - c.agent_radius
        pos = self.pos.astype(np.float64)
        self.n_nudged = 0
        for _ in range(20):
            s, qx, qy, dx, dy, d, ux, uy = self._agent_bar_geometry(pos)
            pen = rmin - d
            inside = pen > 0
            if not inside.any():
                break
            self.n_nudged += int(inside.any(-1).sum())
            # normal: away from the axis; on the axis use the bar's left normal
            on = d < 1e-9
            nx = np.where(on, -uy, dx / np.where(on, 1, d))
            ny = np.where(on, ux, dy / np.where(on, 1, d))
            k = np.argmax(np.where(inside, pen, -np.inf), -1)  # deepest bar
            e, a = np.nonzero(inside.any(-1))
            b = k[e, a]
            step = pen[e, a, b] + 1e-3
            pos[e, a, 0] += nx[e, a, b] * step
            pos[e, a, 1] += ny[e, a, b] * step
            np.clip(pos, lo, hi, out=pos)
        self.pos = pos.astype(np.float32)

    # ------------------------------------------------------------------ stats
    def _init_stats(self):
        E, A, B = self.E, self.A, self.cfg.n_bars
        self.stat_push_agent_steps = np.zeros(E)       # agent-steps pushing any bar
        self.stat_push_bar_steps = np.zeros((E, B))    # steps a bar had >=1 pusher
        self.stat_multi_bar_steps = np.zeros((E, B))   # steps a bar had >=2 pushers
        self.stat_max_pushers = np.zeros((E, B), int)
        self.stat_path = np.zeros((E, B))              # path length of the centre
        self.stat_multi_path = np.zeros((E, B))        # path length while >=2 pushed
        self.stat_turn = np.zeros((E, B))              # total |rotation| (rad)
        self.stat_agent_push = np.zeros((E, A))        # steps each agent pushed
        self.last_npush = np.zeros((E, B), int)        # pushers per bar, last step

    def bar_stats(self):
        """Per-arena bar statistics for the finished episode (dict of arrays)."""
        disp = np.sqrt(((self.bc - self.bc0) ** 2).sum(-1))
        rot = np.abs(self.bphi - self.bphi0)
        moved = disp > 10.0
        return {
            "net_disp": disp,                         # (E,B) final - initial centre
            "net_rot": rot,                           # (E,B) |final - initial angle|
            "path": self.stat_path.copy(),            # (E,B)
            "turn": self.stat_turn.copy(),            # (E,B)
            "bars_moved_10": moved.sum(1),            # (E,) bars with net disp > 10
            "push_agent_steps": self.stat_push_agent_steps.copy(),
            "push_bar_steps": self.stat_push_bar_steps.copy(),
            "multi_bar_steps": self.stat_multi_bar_steps.copy(),
            "multi_path": self.stat_multi_path.copy(),
            "max_pushers": self.stat_max_pushers.copy(),
            "agent_push_steps": self.stat_agent_push.copy(),
        }

    # ------------------------------------------------------------------ step
    def step(self, actions):
        self._move_agents(actions)
        if self.cfg.n_bars:
            self._bars_step()
        return self._eat()

    def _bars_step(self):
        c = self.cfg
        h, rb = c.bar_length / 2, c.bar_radius
        M, I = c.bar_mass, c.bar_inertia
        old_c = self.bc.copy()
        # 1. integrate bars (damped)
        self.bv *= c.bar_damping
        self.bw *= c.bar_ang_damping
        self.bc += self.bv
        self.bphi += self.bw
        self.stat_turn += np.abs(self.bw)
        # 2. bar-bar separation, 3. walls
        self._bar_bar()
        self._bar_walls()
        # 4. agent-bar contacts
        pos = self.pos.astype(np.float64)
        vel = self.vel.astype(np.float64)
        s, qx, qy, dx, dy, d, ux, uy = self._agent_bar_geometry(pos)
        rmin = c.agent_radius + rb
        contact = d < rmin
        npush = np.zeros(self.bw.shape, int)
        if contact.any():
            e, a, b = np.nonzero(contact)
            dd = np.maximum(d[e, a, b], 1e-9)
            nx, ny = dx[e, a, b] / dd, dy[e, a, b] / dd
            # lever arm from the bar centre to the contact point
            rx = qx[e, a, b] - self.bc[e, b, 0]  # + rb*n is parallel to n: adds no torque
            ry = qy[e, a, b] - self.bc[e, b, 1]
            rn = _cross(rx, ry, nx, ny)
            vbx = self.bv[e, b, 0] - self.bw[e, b] * ry
            vby = self.bv[e, b, 1] + self.bw[e, b] * rx
            vrel = (vel[e, a, 0] - vbx) * nx + (vel[e, a, 1] - vby) * ny
            j = np.where(vrel < 0, -vrel / (1.0 + 1.0 / M + rn * rn / I), 0.0)
            # agent (mass 1)
            vel[e, a, 0] += j * nx
            vel[e, a, 1] += j * ny
            # bar: all impulses summed
            np.add.at(self.bv[..., 0], (e, b), -j * nx / M)
            np.add.at(self.bv[..., 1], (e, b), -j * ny / M)
            np.add.at(self.bw, (e, b), -rn * j / I)
            # push statistics
            pushing = j > 0
            np.add.at(npush, (e[pushing], b[pushing]), 1)
            pa = np.zeros(self.pos.shape[:2], bool)
            pa[e[pushing], a[pushing]] = True
            self.stat_agent_push += pa
            self.stat_push_agent_steps += pa.sum(1)
            # position: move the agent fully out (sum over bars it touches)
            pen = rmin - d[e, a, b]
            np.add.at(pos[..., 0], (e, a), nx * pen)
            np.add.at(pos[..., 1], (e, a), ny * pen)
            lo, hi = c.agent_radius, c.world_size - c.agent_radius
            below, above = pos < lo, pos > hi
            np.clip(pos, lo, hi, out=pos)
            vel[below & (vel < 0)] = 0
            vel[above & (vel > 0)] = 0
            # squeezed against a wall: remove the remaining overlap by moving the bar
            s, qx, qy, dx, dy, d, ux, uy = self._agent_bar_geometry(pos)
            res = d < rmin - 1e-6
            if res.any():
                e, a, b = np.nonzero(res)
                dd = np.maximum(d[e, a, b], 1e-9)
                pen = rmin - d[e, a, b]
                np.add.at(self.bc[..., 0], (e, b), -dx[e, a, b] / dd * pen)
                np.add.at(self.bc[..., 1], (e, b), -dy[e, a, b] / dd * pen)
                self._bar_walls()
        self.pos, self.vel = pos.astype(np.float32), vel.astype(np.float32)
        # statistics
        step_len = np.sqrt(((self.bc - old_c) ** 2).sum(-1))
        self.stat_path += step_len
        multi = npush >= 2
        self.stat_push_bar_steps += npush >= 1
        self.stat_multi_bar_steps += multi
        self.stat_multi_path += np.where(multi, step_len, 0.0)
        np.maximum(self.stat_max_pushers, npush, out=self.stat_max_pushers)
        self.last_npush = npush

    def _bar_walls(self):
        c = self.cfg
        h, rb, W = c.bar_length / 2, c.bar_radius, c.world_size
        ext = np.stack([h * np.abs(np.cos(self.bphi)), h * np.abs(np.sin(self.bphi))], -1) + rb
        lo, hi = ext, W - ext
        below, above = self.bc < lo, self.bc > hi
        if below.any() or above.any():
            self.bc = np.minimum(np.maximum(self.bc, lo), hi)
            self.bv[below & (self.bv < 0)] = 0
            self.bv[above & (self.bv > 0)] = 0

    def _bar_bar(self):
        c = self.cfg
        B = c.n_bars
        if B < 2:
            return
        h, rb = c.bar_length / 2, c.bar_radius
        M, I = c.bar_mass, c.bar_inertia
        i, k = np.triu_indices(B, 1)
        ci, ck = self.bc[:, i], self.bc[:, k]                      # (E,P,2)
        near = ((ci - ck) ** 2).sum(-1) < (2 * h + 2 * rb) ** 2
        if not near.any():
            return
        e, p = np.nonzero(near)
        bi, bk = i[p], k[p]
        u = np.stack([np.cos(self.bphi), np.sin(self.bphi)], -1)   # (E,B,2)
        best = None
        # endpoints of bar i against segment k, and endpoints of k against i
        for src, dst, sign in ((bi, bk, 1.0), (bk, bi, -1.0)):
            for end in (-1.0, 1.0):
                pxy = self.bc[e, src] + end * h * u[e, src]
                s, qx, qy, dx, dy, d = _point_segment(
                    pxy[:, 0], pxy[:, 1], self.bc[e, dst, 0], self.bc[e, dst, 1],
                    u[e, dst, 0], u[e, dst, 1], h)
                # normal pointing from bar k to bar i
                cand = (d, sign * dx, sign * dy, (pxy[:, 0] + qx) / 2, (pxy[:, 1] + qy) / 2)
                if best is None:
                    best = list(cand)
                else:
                    m = d < best[0]
                    for t in range(5):
                        best[t] = np.where(m, cand[t], best[t])
        d, dx, dy, px, py = best
        hit = d < 2 * rb
        if not hit.any():
            return
        e, bi, bk = e[hit], bi[hit], bk[hit]
        d, dx, dy, px, py = d[hit], dx[hit], dy[hit], px[hit], py[hit]
        dd = np.maximum(d, 1e-9)
        nx, ny = dx / dd, dy / dd
        rix, riy = px - self.bc[e, bi, 0], py - self.bc[e, bi, 1]
        rkx, rky = px - self.bc[e, bk, 0], py - self.bc[e, bk, 1]
        rin, rkn = _cross(rix, riy, nx, ny), _cross(rkx, rky, nx, ny)
        vix = self.bv[e, bi, 0] - self.bw[e, bi] * riy
        viy = self.bv[e, bi, 1] + self.bw[e, bi] * rix
        vkx = self.bv[e, bk, 0] - self.bw[e, bk] * rky
        vky = self.bv[e, bk, 1] + self.bw[e, bk] * rkx
        vrel = (vix - vkx) * nx + (viy - vky) * ny
        j = np.where(vrel < 0, -vrel / (2.0 / M + rin ** 2 / I + rkn ** 2 / I), 0.0)
        for b, sgn, rn in ((bi, 1.0, rin), (bk, -1.0, rkn)):
            np.add.at(self.bv[..., 0], (e, b), sgn * j * nx / M)
            np.add.at(self.bv[..., 1], (e, b), sgn * j * ny / M)
            np.add.at(self.bw, (e, b), sgn * rn * j / I)
            half = (2 * rb - d) / 2
            np.add.at(self.bc[..., 0], (e, b), sgn * nx * half)
            np.add.at(self.bc[..., 1], (e, b), sgn * ny * half)

    # ------------------------------------------------------------------ sensors
    def observe(self, food_delta=None):
        base = super().observe(food_delta)
        return np.concatenate([base, self._bar_sectors()], -1)

    def _bar_sectors(self):
        c = self.cfg
        E, A = self.pos.shape[:2]
        out = np.zeros((E, A, 8), np.float32)
        if c.n_bars == 0:
            return out
        h, rb, R = c.bar_length / 2, c.bar_radius, c.sensor_range
        K = int(np.ceil(c.bar_length / c.bar_sample_spacing)) + 1
        sk = np.linspace(-h, h, K)
        _, qx, qy, _, _, d, ux, uy = self._agent_bar_geometry()           # (E,A,B)
        e, a, b = np.nonzero(d <= R + rb)                                # bars in range
        if len(e) == 0:
            return out
        ux, uy = ux[e, 0, b], uy[e, 0, b]
        px = self.bc[e, b, 0][:, None] + sk[None, :] * ux[:, None]       # (n,K)
        py = self.bc[e, b, 1][:, None] + sk[None, :] * uy[:, None]
        px = np.concatenate([px, qx[e, a, b][:, None]], 1)               # + exact nearest
        py = np.concatenate([py, qy[e, a, b][:, None]], 1)
        ddx = px - self.pos[e, a, 0][:, None]
        ddy = py - self.pos[e, a, 1][:, None]
        dist = np.sqrt(ddx * ddx + ddy * ddy) - rb
        keep = (dist <= R)
        n_idx, _ = np.nonzero(keep)
        e, a = e[n_idx], a[n_idx]
        ddx, ddy, dist = ddx[keep], ddy[keep], np.maximum(dist[keep], 0.0)
        ang = (np.arctan2(ddy, ddx) - self.theta[e, a]) % (2 * np.pi)
        sec = (ang * (8 / (2 * np.pi))).astype(np.int64) % 8
        flat = out.reshape(-1)
        np.maximum.at(flat, (e * A + a) * 8 + sec, (1.0 - dist / R).astype(np.float32))
        return out

    # ------------------------------------------------------------------ render
    def snapshot_extra(self):
        return {"bars": np.concatenate([self.bc[0], self.bphi[0][:, None]], -1).copy(),
                "multi": self.last_npush[0] >= 2}


def _segment_distance(ax, ay, aphi, bx, by, bphi, h):
    """Exact distance between segment a and segments b (arrays), half-length h."""
    bx, by, bphi = np.atleast_1d(bx), np.atleast_1d(by), np.atleast_1d(bphi)
    uax, uay = np.cos(aphi), np.sin(aphi)
    ubx, uby = np.cos(bphi), np.sin(bphi)
    A0 = (ax - h * uax, ay - h * uay)
    A1 = (ax + h * uax, ay + h * uay)
    B0 = (bx - h * ubx, by - h * uby)
    B1 = (bx + h * ubx, by + h * uby)

    def orient(p, q, r):
        return np.sign(_cross(q[0] - p[0], q[1] - p[1], r[0] - p[0], r[1] - p[1]))

    crossing = ((orient(A0, A1, B0) * orient(A0, A1, B1) < 0)
                & (orient(B0, B1, A0) * orient(B0, B1, A1) < 0))
    d = np.minimum.reduce([
        _point_segment(A0[0], A0[1], bx, by, ubx, uby, h)[5],
        _point_segment(A1[0], A1[1], bx, by, ubx, uby, h)[5],
        _point_segment(B0[0], B0[1], ax, ay, uax, uay, h)[5],
        _point_segment(B1[0], B1[1], ax, ay, uax, uay, h)[5]])
    return np.where(crossing, 0.0, d)


def summarize_bar_stats(st, cfg):
    """Scalar summary of BlockEnv.bar_stats() over all arenas of a batch.

    net_disp_mean   mean distance a bar's centre ended from where it started
    net_disp_max    the largest such distance
    bars_moved_10   mean number of bars per episode moved more than 10 units
    path_mean       mean path length of a bar's centre
    net_rot_mean_deg mean |final - initial angle| (degrees)
    push_frac       fraction of agent-steps in contact with a bar and pushing
                    (contact impulse > 0)
    multi_push_bar_steps   mean per episode: bar-steps with >= 2 simultaneous pushers
    episodes_with_multi    fraction of episodes with any such step
    multi_path_frac        fraction of all bar movement made while >= 2 pushed
    max_pushers            most agents pushing one bar at the same time
    """
    E = st["net_disp"].shape[0]
    if st["net_disp"].size == 0:
        return {k: 0.0 for k in ("net_disp_mean", "net_disp_max", "bars_moved_10",
                                 "path_mean", "net_rot_mean_deg", "push_frac",
                                 "multi_push_bar_steps", "episodes_with_multi",
                                 "multi_path_frac", "max_pushers")}
    path = st["path"].sum()
    return {
        "net_disp_mean": float(st["net_disp"].mean()),
        "net_disp_max": float(st["net_disp"].max()),
        "bars_moved_10": float(st["bars_moved_10"].mean()),
        "path_mean": float(st["path"].mean()),
        "net_rot_mean_deg": float(np.degrees(st["net_rot"]).mean()),
        "push_frac": float(st["push_agent_steps"].sum() / (E * cfg.n_agents * cfg.max_steps)),
        "multi_push_bar_steps": float(st["multi_bar_steps"].sum() / E),
        "episodes_with_multi": float((st["multi_bar_steps"].sum(1) > 0).mean()),
        "multi_path_frac": float(st["multi_path"].sum() / path) if path > 0 else 0.0,
        "max_pushers": int(st["max_pushers"].max()),
    }
