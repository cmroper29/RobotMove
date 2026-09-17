"""URDF kinematics on top of pinocchio: FK, IK and configuration-aware IK solution choice.

Internal joint vectors `q` are in URDF units (rad / m). Controller joint values used by the
robot programs (deg / mm) are converted with `to_urdf` / `to_controller`, which apply the
optional per-joint sign/offset and the FANUC J2/J3 coupling from the config.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pinocchio as pin

from .model import RobotParseError
from .transforms import T_inv, frame_from_spec


@contextlib.contextmanager
def _quiet_stderr():
    """Silence output written straight to file descriptor 2 by native libraries."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):  # e.g. captured streams
        yield
        return
    sys.stderr.flush()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


class RobotModel:
    def __init__(self, robot_cfg: dict, diag=None):
        urdf = robot_cfg.get("urdf")
        if not urdf or not Path(urdf).exists():
            raise RobotParseError(f"URDF file not found: {urdf} (set robot.urdf in the config or pass --urdf)")
        if Path(urdf).suffix.lower() == ".xacro" or "xacro" in Path(urdf).name.lower():
            raise RobotParseError(f"{Path(urdf).name} is a xacro file; convert it to plain URDF first, e.g. "
                                  "'pip install xacro' then 'xacro robot.urdf.xacro > robot.urdf'")
        try:
            with _quiet_stderr():  # the C++ URDF parser prints its own errors
                self.model = pin.buildModelFromUrdf(str(urdf))
        except Exception as exc:  # pinocchio raises plain RuntimeError/ValueError for bad files
            raise RobotParseError(f"could not load URDF {Path(urdf).name}: {str(exc).splitlines()[0]}") from None
        self.data = self.model.createData()
        m = self.model
        if m.nq == 0:
            raise RobotParseError(f"URDF {Path(urdf).name} has no movable joints")
        if m.nq != m.nv:
            raise RobotParseError("URDF contains continuous/planar/floating joints; only revolute and prismatic "
                                  "joints are supported")
        self.nq = m.nq
        self.joint_names = [m.names[i] for i in range(1, m.njoints)]
        kinds = []
        for j in m.joints[1:]:
            short = j.shortname().replace("JointModel", "")
            if j.nq != 1:
                continue
            kinds.append(not short.startswith("P"))
        self.revolute = np.array(kinds, dtype=bool)
        self.scale = np.where(self.revolute, 180.0 / math.pi, 1000.0)  # urdf -> controller units

        tip = robot_cfg.get("tip_link")
        if tip is None:
            for cand in ("tool0", "flange", "ee_link", "tool_flange"):
                if m.existFrame(cand):
                    tip = cand
                    break
            else:
                bodies = [f.name for f in m.frames if f.type == pin.FrameType.BODY]
                tip = bodies[-1]
        if not m.existFrame(tip):
            links = [f.name for f in m.frames if f.type == pin.FrameType.BODY]
            raise RobotParseError(f"tip_link '{tip}' not found in the URDF; links: {', '.join(links)}")
        self.tip_link = tip
        self.fid = m.getFrameId(tip)
        self.tip_offset = frame_from_spec(robot_cfg.get("tip_offset"))
        self.tip_offset_inv = T_inv(self.tip_offset)

        n = self.nq
        self.offsets = np.asarray(robot_cfg.get("joint_offsets") or np.zeros(n), float)
        self.signs = np.asarray(robot_cfg.get("joint_signs") or np.ones(n), float)
        self.j23 = bool(robot_cfg.get("fanuc_j23_coupling"))
        for key, arr in (("joint_offsets", self.offsets), ("joint_signs", self.signs)):
            if len(arr) != n:
                raise RobotParseError(f"robot.{key} has {len(arr)} values but the URDF has {n} joints")

        self.lower = np.array(m.lowerPositionLimit, float)
        self.upper = np.array(m.upperPositionLimit, float)
        bad = ~np.isfinite(self.lower) | ~np.isfinite(self.upper) | (self.lower >= self.upper) | (np.abs(self.upper) > 1e6)
        self.lower[bad], self.upper[bad] = -2 * math.pi, 2 * math.pi

        vel = robot_cfg.get("joint_vel_limits")
        if vel is not None:
            self.vel_limit = np.asarray(vel, float) / self.scale
        else:
            self.vel_limit = np.array(m.velocityLimit, float)
            missing = ~np.isfinite(self.vel_limit) | (self.vel_limit <= 0) | (self.vel_limit > 1e6)
            if missing.any():
                if diag:
                    diag.warn(f"URDF has no velocity limit for joints {np.flatnonzero(missing) + 1}; assuming 180 deg/s")
                self.vel_limit[missing] = math.radians(180.0)
        acc = robot_cfg.get("joint_acc_limits")
        if acc is not None:
            self.acc_limit = np.asarray(acc, float) / self.scale
        else:
            self.acc_limit = self.vel_limit / float(robot_cfg.get("accel_time", 0.4))
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ conventions
    def to_urdf(self, q_ctrl) -> np.ndarray:
        c = np.array(q_ctrl, float)[: self.nq]
        if len(c) < self.nq:
            raise RobotParseError(f"a joint position has {len(c)} values but the URDF has {self.nq} joints")
        if self.j23 and self.nq >= 3:
            c[2] = c[2] + c[1]
        return (c - self.offsets) * self.signs / self.scale

    def to_controller(self, q) -> np.ndarray:
        c = np.asarray(q, float) * self.scale * self.signs + self.offsets
        if self.j23 and self.nq >= 3:
            c = c.copy()
            c[2] = c[2] - c[1]
        return c

    def within_limits(self, q, tol=1e-6) -> bool:
        return bool(np.all(q >= self.lower - tol) and np.all(q <= self.upper + tol))

    # ------------------------------------------------------------------ kinematics
    def fk(self, q) -> np.ndarray:
        """Controller flange pose (mm) in the URDF root frame."""
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, float))
        T = self.data.oMf[self.fid].homogeneous.copy()
        T[:3, 3] *= 1000.0
        return T @ self.tip_offset

    def ik(self, T_flange: np.ndarray, q0, tol: float = 1e-10, max_iter: int = 80) -> tuple[np.ndarray, bool]:
        Tl = T_flange @ self.tip_offset_inv
        target = pin.SE3(Tl[:3, :3].copy(), Tl[:3, 3] / 1000.0)
        model, data, fid = self.model, self.data, self.fid
        q = np.array(q0, float)
        eye = np.eye(6)
        err_norm = np.inf
        for _ in range(max_iter):
            pin.framesForwardKinematics(model, data, q)
            iMd = data.oMf[fid].actInv(target)
            err = pin.log6(iMd).vector
            err_norm = float(err @ err)
            if err_norm < tol * tol:
                return q, True
            J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
            J = -pin.Jlog6(iMd.inverse()) @ J
            lam = 1e-12 + 1e-3 * err_norm
            dq = -J.T @ np.linalg.solve(J @ J.T + lam * eye, err)
            step = np.max(np.abs(dq))
            if step > 0.5:
                dq *= 0.5 / step
            q = pin.integrate(model, q, dq)
        return q, err_norm < 1e-12  # ~1e-3 mm / 1e-3 deg

    def config_mismatch(self, q, hints: dict) -> int:
        if not hints or self.nq < 6:
            return 0
        c = self.to_controller(q)
        eps = 1e-4
        n = 0
        if "kuka_T" in hints:
            T = int(hints["kuka_T"])
            for i in range(6):
                negative = bool((T >> i) & 1)
                if negative and c[i] > eps or not negative and c[i] < -eps:
                    n += 1
        if "abb_cf" in hints:
            cf = hints["abb_cf"]
            for idx, want in ((0, cf[0]), (3, cf[1]), (5, cf[2])):
                if want not in (math.floor((c[idx] - eps) / 90.0), math.floor((c[idx] + eps) / 90.0)):
                    n += 1
        if "fanuc_turns" in hints:
            for idx, want in zip((0, 3, 5), hints["fanuc_turns"]):
                if want not in (math.floor((c[idx] + 180 - eps) / 360.0), math.floor((c[idx] + 180 + eps) / 360.0)):
                    n += 1
        return n

    def ik_best(self, T_flange: np.ndarray, q_prev: Optional[np.ndarray], hints: dict,
                n_random: int = 24) -> tuple[np.ndarray, bool, int]:
        """IK for a joint-move target: prefer solutions matching the programmed configuration,
        then the one closest to the current joints. Returns (q, success, config_mismatch)."""
        seeds = []
        if q_prev is not None:
            seeds.append(q_prev)
            if self.nq >= 6 and self.revolute[3:6].all():
                for s4 in (1, -1):  # wrist flip: J5 -> -J5 with J4, J6 turned half a revolution
                    for s6 in (1, -1):
                        f = q_prev.copy()
                        f[3] += s4 * math.pi
                        f[4] = -f[4]
                        f[5] += s6 * math.pi
                        seeds.append(f)
        ref = q_prev if q_prev is not None else 0.5 * (self.lower + self.upper)
        solutions: list[np.ndarray] = []

        def add(q):
            if self.within_limits(q) and not any(np.allclose(q, s, atol=1e-5) for s in solutions):
                solutions.append(q)

        def consider(seed):
            q, ok = self.ik(T_flange, seed)
            if not ok:
                return
            # the same pose is reached with any revolute joint turned by a full revolution
            variants = [q]
            for j in np.flatnonzero(self.revolute & (self.upper - self.lower > 2 * math.pi)):
                variants += [v + sgn * 2 * math.pi * np.eye(self.nq)[j] for v in variants for sgn in (1, -1)]
            for v in variants:
                add(v)

        for s in seeds:
            consider(s)
        good = [q for q in solutions if self.config_mismatch(q, hints) == 0]
        if not good:
            lo = np.maximum(self.lower, -2 * math.pi)
            hi = np.minimum(self.upper, 2 * math.pi)
            for _ in range(n_random):
                consider(self.rng.uniform(lo, hi))
        if not solutions:
            q, _ = self.ik(T_flange, ref, max_iter=200)
            return q, False, 0
        weights = np.where(self.revolute, 1.0, 0.001)
        best = min(solutions, key=lambda q: (self.config_mismatch(q, hints), float(np.sum(weights * np.abs(q - ref)))))
        return best, True, self.config_mismatch(best, hints)
