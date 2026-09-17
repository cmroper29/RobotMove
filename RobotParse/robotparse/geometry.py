"""Cartesian path geometry.

Every path element is parameterised by `p`, which equals TCP arc length in mm for
elements that move the TCP. Pure re-orientation legs use `p = angle * ORI_MM_PER_DEG`.
Orientation follows the shortest rotation between the programmed poses.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

EPS_LEN = 1e-4  # mm; shorter moves count as "position does not change"
ORI_MM_PER_DEG = 1.0


class DegenerateArc(ValueError):
    pass


def _rotations(R0: Rotation, rotvecs: np.ndarray) -> Rotation:
    return R0 * Rotation.from_rotvec(rotvecs)


class CartLeg:
    """Straight-line interpolation of orientation between two poses (shared by all legs)."""

    def __init__(self, T0: np.ndarray, T1: np.ndarray):
        self.T0, self.T1 = T0, T1
        self.R0 = Rotation.from_matrix(T0[:3, :3])
        self.dR = (self.R0.inv() * Rotation.from_matrix(T1[:3, :3])).as_rotvec()
        self.angle = float(np.degrees(np.linalg.norm(self.dR)))
        self.length = 0.0
        self.plen = 0.0

    def _finish(self) -> None:
        self.plen = self.length if self.length > EPS_LEN else max(self.angle * ORI_MM_PER_DEG, 0.0)

    @property
    def moves_position(self) -> bool:
        return self.length > EPS_LEN

    def rot(self, p) -> Rotation:
        p = np.atleast_1d(np.asarray(p, float))
        frac = p / self.plen if self.plen > 0 else np.zeros_like(p)
        return _rotations(self.R0, np.outer(frac, self.dR))

    def end_T(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.pos(self.plen)[0]
        T[:3, :3] = self.rot(self.plen).as_matrix()[0]
        return T


class LineLeg(CartLeg):
    def __init__(self, T0, T1):
        super().__init__(T0, T1)
        d = T1[:3, 3] - T0[:3, 3]
        self.length = float(np.linalg.norm(d))
        self.dir = d / self.length if self.length > EPS_LEN else np.zeros(3)
        self._finish()

    def pos(self, p):
        p = np.atleast_1d(np.asarray(p, float))
        if not self.moves_position:
            return np.repeat(self.T0[None, :3, 3], len(p), axis=0)
        return self.T0[:3, 3] + np.outer(p, self.dir)

    def tangent(self, p):
        p = np.atleast_1d(np.asarray(p, float))
        return np.repeat(self.dir[None, :], len(p), axis=0)


class ArcLeg(CartLeg):
    """Circular arc from T0 to T1 through (or, for FANUC 'A' runs, avoiding) a reference point."""

    def __init__(self, T0, T1, ref_pos, ref_excluded: bool = False, circ_angle: float | None = None):
        super().__init__(T0, T1)
        a_pt, b_pt, c_pt = T0[:3, 3], T1[:3, 3], np.asarray(ref_pos, float)
        a, b = b_pt - a_pt, c_pt - a_pt
        axb = np.cross(a, b)
        denom = 2.0 * float(axb @ axb)
        if denom < 1e-9 * max(float(a @ a) * float(b @ b), 1e-12):
            raise DegenerateArc("arc points are collinear or coincident")
        self.center = a_pt + np.cross(float(a @ a) * b - float(b @ b) * a, axb) / denom
        self.radius = float(np.linalg.norm(a_pt - self.center))
        n = axb / np.linalg.norm(axb)
        for _ in range(2):
            self.e1 = (a_pt - self.center) / self.radius
            self.e2 = np.cross(n, self.e1)
            th_end, th_ref = self._theta(b_pt), self._theta(c_pt)
            ref_inside = th_ref < th_end
            if ref_inside != ref_excluded:
                break
            n = -n
        sweep = th_end
        if circ_angle is not None:
            sweep = np.radians(abs(circ_angle))
            if circ_angle < 0:
                n = -n
                self.e2 = np.cross(n, self.e1)
        self.sweep = float(sweep)
        self.length = self.radius * self.sweep
        self._finish()

    def _theta(self, x) -> float:
        v = x - self.center
        return float(np.arctan2(v @ self.e2, v @ self.e1) % (2 * np.pi))

    def pos(self, p):
        th = np.atleast_1d(np.asarray(p, float)) / self.radius
        return self.center + self.radius * (np.outer(np.cos(th), self.e1) + np.outer(np.sin(th), self.e2))

    def tangent(self, p):
        th = np.atleast_1d(np.asarray(p, float)) / self.radius
        return -np.outer(np.sin(th), self.e1) + np.outer(np.cos(th), self.e2)


class _ArcLengthCurve:
    """Re-parameterises a smooth curve f(u) by arc length."""

    def __init__(self, f, df, u0: float, u1: float, samples: int):
        u = np.linspace(u0, u1, samples)
        speed = np.linalg.norm(df(u), axis=1)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))])
        self.length = float(s[-1])
        keep = np.concatenate([[True], np.diff(s) > 1e-12])
        self.u_of_s = CubicSpline(s[keep], u[keep])
        self.f, self.df = f, df

    def u(self, s):
        return self.u_of_s(np.clip(s, 0.0, self.length))

    def pos(self, s):
        return self.f(self.u(s))

    def tangent(self, s):
        d = self.df(self.u(s))
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


class SplineLeg(CartLeg):
    """Smooth spline through consecutive KUKA SPL points (C2 position, C2 orientation)."""

    def __init__(self, T_start, targets: list[np.ndarray]):
        poses = [T_start] + list(targets)
        super().__init__(T_start, poses[-1])
        pts = np.array([T[:3, 3] for T in poses])
        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], chords > EPS_LEN])
        pts = pts[keep]
        rots = Rotation.from_matrix(np.array([T[:3, :3] for T, k in zip(poses, keep) if k]))
        if len(pts) < 3:
            raise DegenerateArc("spline needs at least two distinct points after the start")
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        spline = CubicSpline(u, pts, bc_type="natural")
        dspline = spline.derivative()
        self.curve = _ArcLengthCurve(spline, dspline, 0.0, u[-1], samples=max(64, 40 * len(pts)))
        self.rot_spline = RotationSpline(u, rots)
        self.u_end = u[-1]
        self.length = self.curve.length
        self._finish()

    def pos(self, p):
        return self.curve.pos(np.atleast_1d(np.asarray(p, float)))

    def tangent(self, p):
        return self.curve.tangent(np.atleast_1d(np.asarray(p, float)))

    def rot(self, p) -> Rotation:
        p = np.atleast_1d(np.asarray(p, float))
        inside = np.clip(p, 0.0, self.plen)
        R = self.rot_spline(self.curve.u(inside))
        over = p - inside
        if np.any(over != 0):  # extrapolate with the end angular velocity (used by corner blends)
            h = min(1.0, 0.01 * self.plen)
            ends = np.where(over > 0, self.plen, 0.0)
            inner = np.where(over > 0, self.plen - h, h)
            Ra, Rb = self.rot_spline(self.curve.u(inner)), self.rot_spline(self.curve.u(ends))
            rate = (Ra.inv() * Rb).as_rotvec() / h  # local rotation per mm pointing outwards
            R = R * Rotation.from_rotvec(rate * np.abs(over)[:, None])
        return R


class LegPiece:
    """The part [p0, p1] of a leg that is not consumed by corner blends."""

    kind = "leg"

    def __init__(self, leg, p0: float, p1: float, index: int):
        self.leg, self.p0, self.index = leg, p0, index
        self.length = p1 - p0

    def pos(self, p):
        return self.leg.pos(self.p0 + np.asarray(p))

    def rot(self, p) -> Rotation:
        return self.leg.rot(self.p0 + np.asarray(p))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class BlendPiece:
    """Tangent-continuous corner blend between leg A (from pA to its end) and leg B (from 0 to pB).

    Position: cubic Bezier whose control legs follow the leg tangents (a parabola for two lines,
    the usual zone/approximation shape). Orientation: a smoothstep mix of both legs' orientation
    functions, so the angular rate is continuous at both ends of the blend.
    """

    kind = "blend"

    def __init__(self, legA, pA: float, legB, pB: float, index: int):
        self.legA, self.pA, self.legB, self.pB, self.index = legA, pA, legB, pB, index
        dA, dB = legA.plen - pA, pB
        P0, t0 = legA.pos(pA)[0], legA.tangent(pA)[0]
        P3, t3 = legB.pos(pB)[0], legB.tangent(pB)[0]
        ctrl = np.array([P0, P0 + t0 * dA * 2.0 / 3.0, P3 - t3 * dB * 2.0 / 3.0, P3])

        def f(u):
            u = np.atleast_1d(u)[:, None]
            v = 1.0 - u
            return v**3 * ctrl[0] + 3 * v * v * u * ctrl[1] + 3 * v * u * u * ctrl[2] + u**3 * ctrl[3]

        def df(u):
            u = np.atleast_1d(u)[:, None]
            v = 1.0 - u
            return 3 * v * v * (ctrl[1] - ctrl[0]) + 6 * v * u * (ctrl[2] - ctrl[1]) + 3 * u * u * (ctrl[3] - ctrl[2])

        self.curve = _ArcLengthCurve(f, df, 0.0, 1.0, samples=256)
        self.length = self.curve.length

    def pos(self, p):
        return self.curve.pos(np.atleast_1d(np.asarray(p, float)))

    def rot(self, p) -> Rotation:
        p = np.atleast_1d(np.asarray(p, float))
        L = max(self.length, 1e-12)
        RA = self.legA.rot(self.pA + p)
        RB = self.legB.rot(self.pB - L + p)
        w = smoothstep(p / L)
        return RA * Rotation.from_rotvec((RA.inv() * RB).as_rotvec() * w[:, None])


class PiecewisePath:
    """Concatenation of pieces with a global path parameter."""

    def __init__(self, pieces: list):
        self.pieces = pieces
        lengths = np.array([pc.length for pc in pieces])
        self.starts = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(self.starts[-1])

    def locate(self, p):
        p = np.atleast_1d(np.asarray(p, float))
        idx = np.clip(np.searchsorted(self.starts, p, side="right") - 1, 0, len(self.pieces) - 1)
        return idx, p - self.starts[idx]

    def evaluate(self, p) -> tuple[np.ndarray, np.ndarray]:
        """Positions (N,3) and rotation matrices (N,3,3); p outside the path is extrapolated."""
        idx, local = self.locate(p)
        pos = np.empty((len(idx), 3))
        rot = np.empty((len(idx), 3, 3))
        for i in np.unique(idx):
            sel = idx == i
            pc = self.pieces[i]
            pos[sel] = pc.pos(local[sel])
            rot[sel] = pc.rot(local[sel]).as_matrix()
        return pos, rot
