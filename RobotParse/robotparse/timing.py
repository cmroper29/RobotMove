"""Time-optimal path parameterisation with TOPP-RA (toppra).

Every limit is expressed as a box constraint on a path derivative evaluated at grid points:

* velocity rows  v'(s) * sdot           within +-vel_limit
* accel rows     a'(s) * sddot + a''(s) * sdot^2   within +-acc_limit

Programmed speeds are encoded as a velocity row with derivative 1/v_max(s) and limit 1, and
varying acceleration limits by dividing the rows by their scale, so constant toppra
constraints handle all of them.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .model import RobotParseError

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import toppra
    import toppra.algorithm as ta_algo
    import toppra.constraint as ta_constraint

logging.getLogger("toppra").setLevel(logging.ERROR)

S_SCALE = 1000.0  # toppra works in path units of S_SCALE mm (metres) for better conditioning
BIG = 1e9


class PlanningError(RobotParseError):
    pass


class GridPath(toppra.interpolator.AbstractGeometricPath):
    """A geometric path known only through derivative samples at grid points."""

    def __init__(self, grid: np.ndarray, d1: np.ndarray, d2: np.ndarray):
        self.grid, self.d1, self.d2 = grid, d1, d2

    @property
    def dof(self) -> int:
        return self.d1.shape[1]

    @property
    def path_interval(self):
        return np.array([self.grid[0], self.grid[-1]])

    def __call__(self, s, order: int = 0):
        arr = np.atleast_1d(np.asarray(s, float))
        if order == 0:
            out = np.zeros((len(arr), self.dof))
            out[:, -1] = arr
        else:
            src = self.d1 if order == 1 else self.d2
            idx = np.clip(np.searchsorted(self.grid, arr), 0, len(self.grid) - 1)
            prev = np.clip(idx - 1, 0, len(self.grid) - 1)
            closer = np.abs(self.grid[prev] - arr) < np.abs(self.grid[idx] - arr)
            out = src[np.where(closer, prev, idx)]
        return out if np.ndim(s) else out[0]


class TimedPath:
    """Result of parameterisation: maps time to the path parameter p (mm)."""

    def __init__(self, traj, length: float):
        self._traj = traj
        self.length = length
        self.duration = float(traj.duration) if traj is not None else 0.0

    def p_at(self, t: np.ndarray) -> np.ndarray:
        if self._traj is None:
            return np.zeros_like(t)
        t = np.clip(t, 0.0, self.duration)
        return np.clip(self._traj.eval(t)[:, -1] * S_SCALE, 0.0, self.length)


def parameterize(grid_p: np.ndarray, vmax_p: np.ndarray, vel_rows: np.ndarray, vel_limits: np.ndarray,
                 acc_d1: np.ndarray, acc_d2: np.ndarray, acc_limits: np.ndarray) -> TimedPath:
    """Time-optimal rest-to-rest timing along a path.

    grid_p     (N,)   grid in mm, strictly increasing from 0
    vmax_p     (N,)   programmed path-speed limit (mm/s) at each grid point
    vel_rows   (N,kv) first derivatives (per mm) of velocity-limited quantities
    vel_limits (kv,)
    acc_d1/d2  (N,ka) first/second derivatives (per mm, per mm^2) of acceleration-limited quantities
    acc_limits (ka,)
    """
    length = float(grid_p[-1])
    if length <= 0:
        return TimedPath(None, 0.0)
    n = len(grid_p)
    s = grid_p / S_SCALE
    kv, ka = vel_rows.shape[1], acc_d1.shape[1]
    speed_row = (S_SCALE / np.maximum(vmax_p, 1e-6))[:, None]
    d1 = np.hstack([vel_rows * S_SCALE, acc_d1 * S_SCALE, speed_row])
    d2 = np.hstack([np.zeros((n, kv)), acc_d2 * S_SCALE**2, np.zeros((n, 1))])
    d1[np.abs(d1) < 1e-12] = 1e-12
    vlim = np.concatenate([vel_limits, np.full(ka, BIG), [1.0]])
    alim = np.concatenate([np.full(kv, BIG), acc_limits, [BIG]])
    path = GridPath(s, d1, d2)
    constraints = [
        ta_constraint.JointVelocityConstraint(np.column_stack([-vlim, vlim])),
        ta_constraint.JointAccelerationConstraint(np.column_stack([-alim, alim])),
    ]
    instance = ta_algo.TOPPRA(constraints, path, gridpoints=s, parametrizer="ParametrizeConstAccel")
    traj = instance.compute_trajectory(0.0, 0.0)
    if traj is None:
        raise PlanningError("toppra could not time-parameterise the path (infeasible limits)")
    return TimedPath(traj, length)


def make_grid(piece_lengths: list[float], step: float, min_per_piece: list[int]) -> np.ndarray:
    """Grid over concatenated pieces, including every piece boundary."""
    pts = [np.array([0.0])]
    start = 0.0
    for L, nmin in zip(piece_lengths, min_per_piece):
        if L <= 0:
            continue
        n = max(int(np.ceil(L / step)), nmin, 1)
        pts.append(start + np.linspace(0.0, L, n + 1)[1:])
        start += L
    grid = np.concatenate(pts)
    keep = np.concatenate([[True], np.diff(grid) > 1e-3])  # merge near-duplicates (bad for derivatives and LPs)
    keep[-1] = True
    grid = grid[keep]
    if len(grid) > 2 and grid[-1] - grid[-2] <= 1e-3:
        grid = np.delete(grid, -2)
    return grid
