"""Planning: parsed program -> time-sampled TCP trajectory.

dumb  : Cartesian estimate. Legs are lines/arcs/splines in Cartesian space (joint moves are
        approximated as straight lines), corners are blended using the programmed zones, and
        the path is timed with programmed TCP/orientation speeds plus a Cartesian
        acceleration limit.
smart : URDF based. Joint moves interpolate in joint space (IK picks the programmed
        configuration), Cartesian moves are tracked with IK, zones blend in Cartesian space
        (CP-CP) or joint space (any corner touching a joint move), and the path is timed with
        joint velocity/acceleration limits as well as programmed speeds. TCP = FK(joints).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from .geometry import EPS_LEN, ArcLeg, BlendPiece, DegenerateArc, LegPiece, LineLeg, PiecewisePath, SplineLeg
from .model import CartTarget, Diagnostics, Dwell, JointTarget, Motion, Program, RelativeTarget, Source
from .timing import PlanningError, TimedPath, make_grid, parameterize
from .transforms import T_inv, frame_from_spec, rotation_angle_deg

MIN_BLEND = 0.01  # mm; smaller zones are treated as no blend
MIN_PIECE = 1e-4  # mm; leftovers of a leg shorter than this (zones meeting mid-segment) are dropped
FD_STEP_DUMB = 0.05  # mm; finite-difference step for path derivatives (positions are exact)
TANGENT_TOL_DEG = 0.5
JOINT_MM_PER_RAD = 300.0  # path-parameter length of joint moves per radian of the largest joint change
MAX_GRID_POINTS = 400_000


@dataclass
class TrajectoryResult:
    t: np.ndarray
    position: np.ndarray  # (N,3) mm in the robot base frame
    rotation: np.ndarray  # (N,3,3)
    joints: Optional[np.ndarray] = None  # (N,nq) controller units
    joint_names: list = field(default_factory=list)
    source: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class Leg:
    motion: Motion
    tool: np.ndarray
    geom: object = None  # Cartesian geometry; None for joint-space legs (smart mode)
    T0: Optional[np.ndarray] = None
    T1: Optional[np.ndarray] = None
    q0: Optional[np.ndarray] = None
    q1: Optional[np.ndarray] = None
    v_tcp: float = math.inf
    v_ori: float = math.inf
    joint_frac: Optional[float] = None
    joint_rate: Optional[float] = None
    accel_scale: float = 1.0
    cart_accel: float = 2500.0

    @property
    def is_joint(self) -> bool:
        return self.geom is None

    @property
    def length(self) -> float:
        return self.geom.length if self.geom is not None else float(np.linalg.norm(self.T1[:3, 3] - self.T0[:3, 3]))

    @property
    def angle(self) -> float:
        if self.geom is not None:
            return self.geom.angle
        return rotation_angle_deg(self.T0[:3, :3], self.T1[:3, :3])

    @property
    def plen(self) -> float:
        if self.geom is not None:
            return self.geom.plen
        return max(JOINT_MM_PER_RAD * float(np.max(np.abs(self.q1 - self.q0))), 1e-9)

    @property
    def moves_position(self) -> bool:
        return self.length > EPS_LEN

    @property
    def source(self) -> Source:
        return self.motion.source


@dataclass
class DwellItem:
    duration: float
    source: Source


# =============================================================================== resolution
class _Resolver:
    def __init__(self, program: Program, cfg: dict, diag: Diagnostics, robot=None, smart: bool = False):
        self.program, self.cfg, self.diag, self.robot, self.smart = program, cfg, diag, robot, smart
        self.mc = cfg["motion"]
        self.T_cur: Optional[np.ndarray] = None  # TCP pose (with tool_cur)
        self.tool_cur: Optional[np.ndarray] = None
        self.q_cur: Optional[np.ndarray] = None  # URDF joints (smart, or dumb after a joint target)
        init = cfg["robot"].get("initial_joints")
        if init is not None and robot is not None:
            self.q_cur = robot.to_urdf(init)

    # -- target helpers -----------------------------------------------------
    def _cannot(self, where, reason: str, is_via: bool) -> None:
        if is_via:
            self.diag.warn(f"{where}: circle point: {reason}; moving linearly instead")
        else:
            self.diag.drop(where, reason)

    def cart_pose(self, target, m: Motion, is_via: bool = False) -> Optional[np.ndarray]:
        where = m.source
        if isinstance(target, CartTarget):
            return m.base @ target.T
        if isinstance(target, JointTarget):
            if self.robot is None:
                self.diag.need("urdf", "robot model", where)
                self._cannot(where, "a joint position needs a robot model (URDF) to compute the TCP", is_via)
                return None
            return self.robot.fk(self.robot.to_urdf(target.q)) @ m.tool
        if isinstance(target, RelativeTarget):
            if target.joints is not None:
                if self.robot is None or self.q_cur is None:
                    if self.robot is None:
                        self.diag.need("urdf", "robot model", where)
                    self._cannot(where, "a relative joint move needs a robot model (URDF) and a known start", is_via)
                    return None
                q = self.robot.to_urdf(self.robot.to_controller(self.q_cur) + target.joints)
                return self.robot.fk(q) @ m.tool
            T = self.current_tcp(m.tool)
            if T is None:
                self._cannot(where, "a relative move needs a known start position", is_via)
                return None
            if target.frame == "tool":
                return T @ target.delta
            Tb = T_inv(m.base) @ T
            new = Tb.copy()
            new[:3, 3] += target.delta[:3, 3]
            new[:3, :3] = target.delta[:3, :3] @ Tb[:3, :3]
            return m.base @ new
        return None

    def current_tcp(self, tool: np.ndarray) -> Optional[np.ndarray]:
        if self.smart and self.q_cur is not None:
            return self.robot.fk(self.q_cur) @ tool
        if self.T_cur is None:
            if self.q_cur is not None and self.robot is not None:
                return self.robot.fk(self.q_cur) @ tool
            return None
        if self.tool_cur is not None and not np.allclose(self.tool_cur, tool):
            return self.T_cur @ T_inv(self.tool_cur) @ tool  # same flange, different TCP
        return self.T_cur

    def joint_goal(self, m: Motion) -> Optional[np.ndarray]:
        robot, target = self.robot, m.target
        if isinstance(target, JointTarget):
            return robot.to_urdf(target.q)
        if isinstance(target, RelativeTarget) and target.joints is not None:
            if self.q_cur is None:
                self.diag.drop(m.source, "a relative joint move needs a known start position")
                return None
            return robot.to_urdf(robot.to_controller(self.q_cur) + target.joints)
        T = self.cart_pose(target, m)
        if T is None:
            return None
        hints = target.config if isinstance(target, CartTarget) else {}
        q, ok, mismatch = robot.ik_best(T @ T_inv(m.tool), self.q_cur, hints,
                                        n_random=int(self.cfg["robot"]["ik_random_seeds"]))
        if not ok:
            self.diag.issue("unreachable", m.source)
            self.diag.warn(f"{m.source}: target is unreachable for this URDF (IK failed); using the closest pose")
        elif mismatch:
            self.diag.issue("config_mismatch", m.source)
            self.diag.warn(f"{m.source}: no IK solution matches the programmed configuration; using the closest one")
        return q

    # -- speeds -----------------------------------------------------------------
    def apply_speed(self, leg: Leg) -> None:
        sp, mc = leg.motion.speed, self.mc
        leg.accel_scale = max(sp.accel_pct, 1.0) / 100.0
        leg.cart_accel = (sp.cart_accel or mc["cart_accel"]) * leg.accel_scale
        joint_move = leg.motion.kind == "PTP"
        if sp.duration:
            T = max(sp.duration, 1e-3)
            if leg.is_joint:
                leg.joint_rate = leg.plen / T
            else:
                leg.v_tcp = leg.length / T if leg.moves_position else math.inf
                leg.v_ori = leg.angle / T if leg.angle > 1e-6 else math.inf
                if not math.isfinite(leg.v_tcp) and not math.isfinite(leg.v_ori):
                    leg.v_tcp = mc["default_tcp_speed"]
        elif joint_move and sp.tcp is None:
            frac = (sp.joint_pct if sp.joint_pct is not None else 100.0) / 100.0
            if leg.is_joint:
                leg.joint_frac = frac
            else:
                leg.v_tcp, leg.v_ori = frac * mc["ptp_tcp_speed"], frac * mc["ptp_ori_speed"]
        else:
            v_tcp = sp.tcp or mc["default_tcp_speed"]
            v_ori = min(sp.ori or mc["max_ori_speed"], mc["max_ori_speed"])
            if leg.is_joint:  # e.g. ABB MoveJ: speed data approximates the TCP speed
                t = max(leg.length / v_tcp, leg.angle / v_ori, 1e-6)
                leg.joint_rate = leg.plen / t
            else:
                leg.v_tcp, leg.v_ori = v_tcp, v_ori

    def nominal_tcp_speed(self, leg: Leg) -> float:
        if math.isfinite(leg.v_tcp):
            return leg.v_tcp
        if leg.joint_frac is not None:
            return leg.joint_frac * self.mc["ptp_tcp_speed"]
        return self.mc["default_tcp_speed"]

    # -- main -----------------------------------------------------------------
    def resolve(self) -> list:
        items: list = []
        cmds = self.program.commands
        i = 0
        while i < len(cmds):
            c = cmds[i]
            if isinstance(c, Dwell):
                items.append(DwellItem(c.duration, c.source))
                i += 1
                continue
            if c.kind == "SPLINE":
                j = i
                while j < len(cmds) and isinstance(cmds[j], Motion) and cmds[j].kind == "SPLINE":
                    j += 1
                self._spline(cmds[i:j], items)
                i = j
                continue
            if self.smart and c.kind == "PTP":
                self._joint_leg(c, items)
            else:
                self._cart_leg(c, items)
            i += 1
        return items

    def _start_here(self, m: Motion, T: np.ndarray) -> bool:
        """The first reachable target defines where the robot starts."""
        if self.smart:
            if self.q_cur is None:
                self.q_cur = self._ik_track_start(T, m)
                return True
            return False
        if self.T_cur is None and self.q_cur is None:
            self.T_cur, self.tool_cur = T, m.tool
            return True
        return False

    def _ik_track_start(self, T, m):
        hints = m.target.config if isinstance(m.target, CartTarget) else {}
        q, ok, _ = self.robot.ik_best(T @ T_inv(m.tool), None, hints, int(self.cfg["robot"]["ik_random_seeds"]))
        if not ok:
            self.diag.issue("unreachable", m.source)
            self.diag.warn(f"{m.source}: start pose is unreachable for this URDF")
        return q

    def _joint_leg(self, m: Motion, items: list) -> None:
        q_goal = self.joint_goal(m)
        if q_goal is None:
            return
        if self.q_cur is None:
            self.q_cur = q_goal
            return
        if np.max(np.abs(q_goal - self.q_cur)) < 1e-9:
            return
        leg = Leg(m, m.tool, None, self.robot.fk(self.q_cur) @ m.tool, self.robot.fk(q_goal) @ m.tool,
                  self.q_cur.copy(), q_goal)
        self.apply_speed(leg)
        items.append(leg)
        self.q_cur = q_goal

    def _cart_leg(self, m: Motion, items: list) -> None:
        T_goal = self.cart_pose(m.target, m)
        if T_goal is None:
            return
        joint_goal = self.robot.to_urdf(m.target.q) if isinstance(m.target, JointTarget) and self.robot else None
        start = self.current_tcp(m.tool)
        if start is None:
            if self.smart:
                self.q_cur = joint_goal if joint_goal is not None else self._ik_track_start(T_goal, m)
            else:
                self.T_cur, self.tool_cur, self.q_cur = T_goal, m.tool, joint_goal
            return
        geom = None
        if m.kind == "CIRC" and m.via is not None:
            via = self.cart_pose(m.via, m, is_via=True)
            if via is not None:
                try:
                    geom = ArcLeg(start, T_goal, via[:3, 3], m.arc_ref_excluded, m.circ_angle)
                except DegenerateArc:
                    self.diag.warn(f"{m.source}: circle points are collinear; moving linearly")
        if geom is None:
            geom = LineLeg(start, T_goal)
        self._push_cart(m, geom, start, items)
        if not self.smart and joint_goal is not None:
            self.q_cur = joint_goal  # dumb mode: remember joints so relative joint moves still work

    def _spline(self, motions: list[Motion], items: list) -> None:
        poses, last = [], motions[-1]
        for m in motions:
            T = self.cart_pose(m.target, m)
            if T is None:
                continue
            if self.current_tcp(m.tool) is None and not poses:
                self._start_here(m, T)
                continue
            poses.append(T)
        start = self.current_tcp(last.tool)
        if not poses or start is None:
            return
        try:
            geom = SplineLeg(start, poses)
        except DegenerateArc:
            for T in poses:  # fall back to straight segments
                s = self.current_tcp(last.tool)
                self._push_cart(last, LineLeg(s, T), s, items)
            return
        self._push_cart(last, geom, start, items)

    def _push_cart(self, m: Motion, geom, start: np.ndarray, items: list) -> None:
        if geom.plen <= 1e-9:
            return
        leg = Leg(m, m.tool, geom, start, geom.end_T())
        if self.smart:
            leg.q0 = self.q_cur.copy()
            leg.q1 = self._track(geom, m.tool, self.q_cur, m.source)
            self.q_cur = leg.q1
        else:
            self.T_cur, self.tool_cur = leg.T1, m.tool
            if isinstance(m.target, CartTarget):
                self.q_cur = None
        self.apply_speed(leg)
        items.append(leg)

    def _track(self, geom, tool, q, source) -> np.ndarray:
        """Coarse IK along a Cartesian leg to find the joint state at its end."""
        tool_inv = T_inv(tool)
        n = max(2, int(math.ceil(geom.plen / 10.0)))
        ps = np.linspace(0.0, geom.plen, n + 1)[1:]
        pos, rot = geom.pos(ps), geom.rot(ps).as_matrix()
        for k in range(len(ps)):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rot[k], pos[k]
            q, ok = self.robot.ik(T @ tool_inv, q)
            if not ok:
                self.diag.issue("unreachable", source)
                self.diag.warn(f"{source}: Cartesian path leaves the reachable workspace (IK failed)", key=f"track:{source}")
        return q


# =============================================================================== runs
@dataclass
class Run:
    legs: list
    corners: list  # blend distance (mm) between legs i and i+1


def corner_distance(A: Leg, B: Leg, resolver: _Resolver) -> Optional[float]:
    """Blend distance at the corner between two legs, or None when the robot must stop there."""
    z = A.motion.zone
    if z.fine or not (A.moves_position and B.moves_position):
        return None
    if not np.allclose(A.tool, B.tool):
        return None  # blending across a tool change would mix two different TCPs
    LA, LB = A.length, B.length
    if z.dist is not None:
        d = z.dist
    elif z.pct is not None:
        d = z.pct / 100.0 * 0.5 * min(LA, LB)
    elif z.cnt is not None:
        v = min(resolver.nominal_tcp_speed(A), resolver.nominal_tcp_speed(B))
        d = z.cnt / 100.0 * v * resolver.mc["fanuc_cnt_time"]
    else:
        d = 0.0
    d = min(d, 0.5 * LA, 0.5 * LB)
    if d >= MIN_BLEND:
        return d
    if not A.is_joint and not B.is_joint:
        ta, tb = A.geom.tangent(A.geom.plen)[0], B.geom.tangent(0.0)[0]
        if np.degrees(np.arccos(np.clip(ta @ tb, -1, 1))) < TANGENT_TOL_DEG:
            return 0.0
    return None


def split_runs(items: list, resolver: _Resolver) -> list:
    out, legs, corners = [], [], []
    for i, it in enumerate(items):
        if isinstance(it, DwellItem):
            if legs:
                out.append(Run(legs, corners))
                legs, corners = [], []
            out.append(it)
            continue
        legs.append(it)
        nxt = items[i + 1] if i + 1 < len(items) else None
        d = corner_distance(it, nxt, resolver) if isinstance(nxt, Leg) else None
        if d is None:
            out.append(Run(legs, corners))
            legs, corners = [], []
        else:
            corners.append(d)
    if legs:
        out.append(Run(legs, corners))
    return out


# =============================================================================== smart pieces
class JointLegPiece:
    kind = "joint"

    def __init__(self, leg: Leg, p0: float, p1: float, index: int):
        self.q0, self.dq = leg.q0, (leg.q1 - leg.q0) / leg.plen
        self.p0, self.length, self.index, self.tool = p0, p1 - p0, index, leg.tool

    def q(self, p):
        return self.q0 + np.outer(self.p0 + np.atleast_1d(p), self.dq)


class JointBlendPiece:
    kind = "joint"

    def __init__(self, qa, ma, qb, mb, length: float, index: int, tool):
        self.qa, self.ma, self.qb, self.mb = qa, ma, qb, mb
        self.length, self.index, self.tool = length, index, tool

    def q(self, p):
        u = np.clip(np.atleast_1d(p) / self.length, 0.0, 1.0)[:, None]
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        L = self.length
        return h00 * self.qa + h10 * L * self.ma + h01 * self.qb + h11 * L * self.mb


class CartPieceSmart:
    kind = "cart"

    def __init__(self, piece, tool, index: int):
        self.piece, self.tool, self.index, self.length = piece, tool, index, piece.length
        self.tool_inv = T_inv(tool)


# =============================================================================== planning
class Planner:
    def __init__(self, program: Program, cfg: dict, mode: str = "dumb", robot=None, diag: Optional[Diagnostics] = None):
        if mode not in ("dumb", "smart"):
            raise ValueError("mode must be 'dumb' or 'smart'")
        if mode == "smart" and robot is None:
            raise ValueError("smart mode needs a URDF robot model")
        self.program, self.cfg, self.mode, self.robot = program, cfg, mode, robot
        self.diag = diag or program.diag or Diagnostics()
        if self.diag is not program.diag:
            for w in list(program.warnings):
                if w not in self.diag.messages:
                    self.diag.warn(w)
        self.resolver = _Resolver(program, cfg, self.diag, robot, smart=(mode == "smart"))
        self.output_tool = resolve_output_tool(cfg)

    # -- public -----------------------------------------------------------------
    def run(self, dt: float) -> TrajectoryResult:
        if dt <= 0:
            raise ValueError("dt must be positive")
        items = self.resolver.resolve()
        if self.output_tool is None:
            legs = [it for it in items if isinstance(it, Leg)]
            for a, b in zip(legs, legs[1:]):
                if not np.allclose(a.tool, b.tool):
                    self.diag.counts["tool_change"] += 1
                    self.diag.warn(f"{b.source}: the active tool changes here, so the reported TCP jumps; set "
                                   "output_tool in the config to report one fixed TCP", key="toolchange")
        if self.mode == "dumb":
            self.diag.counts["dumb_joint_moves"] += sum(1 for it in items if isinstance(it, Leg) and it.motion.kind == "PTP")
        blocks = split_runs(items, self.resolver)
        timeline = []  # (duration, evaluator | None for hold)
        for b in blocks:
            if isinstance(b, DwellItem):
                timeline.append((b.duration, None, b.source))
                continue
            ev = self._plan_run(b)
            if ev is not None:
                timeline.append((ev.duration, ev, None))
        return self._sample(timeline, dt)

    # -- runs ---------------------------------------------------------------------
    def _plan_run(self, run: Run):
        try:
            if self.mode == "dumb":
                return self._plan_dumb(run)
            return self._plan_smart(run)
        except PlanningError as exc:
            srcs = ", ".join(str(l.source) for l in run.legs[:3])
            raise PlanningError(f"{exc} (motions at {srcs})") from exc

    def _cart_pieces(self, run: Run) -> list:
        pieces = []
        for i, leg in enumerate(run.legs):
            start = run.corners[i - 1] if i > 0 else 0.0
            end = leg.geom.plen - (run.corners[i] if i < len(run.corners) else 0.0)
            if end - start > MIN_PIECE:
                pieces.append(LegPiece(leg.geom, start, end, i))
            if i < len(run.corners) and run.corners[i] > 0:
                pieces.append(BlendPiece(leg.geom, leg.geom.plen - run.corners[i], run.legs[i + 1].geom,
                                         run.corners[i], i))
        return pieces

    def _grid(self, pieces, step) -> np.ndarray:
        total = sum(pc.length for pc in pieces)
        step = max(step, total / MAX_GRID_POINTS)
        mins = []
        for pc in pieces:
            curved = not (isinstance(pc, LegPiece) and isinstance(pc.leg, LineLeg)) and not isinstance(pc, JointLegPiece)
            # curved pieces get at least one grid point per step/8 of length (max 16, min 2)
            mins.append(max(2, min(16, int(np.ceil(pc.length / (step / 8))))) if curved else 2)
        return make_grid([pc.length for pc in pieces], step, mins)

    @staticmethod
    def _blend_weight(pc, local):
        if getattr(pc, "kind", "") == "blend" or isinstance(pc, (BlendPiece, JointBlendPiece)) or (
                isinstance(pc, CartPieceSmart) and isinstance(pc.piece, BlendPiece)):
            return np.clip(local / max(pc.length, 1e-12), 0.0, 1.0)
        return None

    def _plan_dumb(self, run: Run):
        path = PiecewisePath(self._cart_pieces(run))
        if path.length <= 0:
            return None
        grid = self._grid(path.pieces, float(self.cfg["motion"]["grid_step"]))
        h = min(FD_STEP_DUMB, 0.25 * path.length)
        c0, c1, c2, ori_rate = _cart_fd(path.evaluate, grid, h, path.length)
        idx, local = path.locate(grid)
        vmax = np.empty(len(grid))
        acc = np.empty(len(grid))
        speed = np.linalg.norm(c1, axis=1)
        for k in range(len(grid)):
            pc = path.pieces[idx[k]]
            legs = run.legs
            w = self._blend_weight(pc, local[k])
            limA = _cart_limit(legs[pc.index], speed[k], ori_rate[k])
            aA = legs[pc.index].cart_accel
            if w is None:
                vmax[k], acc[k] = limA, aA
            else:
                B = legs[pc.index + 1]
                vmax[k] = (1 - w) * limA + w * _cart_limit(B, speed[k], ori_rate[k])
                acc[k] = (1 - w) * aA + w * B.cart_accel
        timed = parameterize(grid, vmax, np.zeros((len(grid), 0)), np.zeros(0),
                             c1 / acc[:, None], c2 / acc[:, None], np.ones(3))
        return _DumbEvaluator(path, timed, run, self.output_tool)

    def _plan_smart(self, run: Run):
        robot = self.robot
        legs, corners = run.legs, run.corners
        h_fd = min(0.5, 0.25 * float(self.cfg["robot"]["grid_step"]))
        pieces = []
        cut_start = [0.0] * len(legs)  # p where each leg's own piece starts
        cut_end = [leg.plen for leg in legs]
        blends: dict[int, object] = {}
        for i, d in enumerate(corners):
            A, B = legs[i], legs[i + 1]
            if d <= 0:
                continue
            if not A.is_joint and not B.is_joint:
                cut_end[i], cut_start[i + 1] = A.geom.plen - d, d
                blends[i] = CartPieceSmart(BlendPiece(A.geom, A.geom.plen - d, B.geom, d, i), B.tool, i)
                continue
            dA = A.plen * d / A.length if A.is_joint else d
            dB = B.plen * d / B.length if B.is_joint else d
            cut_end[i], cut_start[i + 1] = A.plen - dA, dB
            qa, ma = self._leg_q_and_rate(A, A.plen - dA, A.q1, h_fd)
            qb, mb = self._leg_q_and_rate(B, dB, B.q0, h_fd)
            blends[i] = JointBlendPiece(qa, ma, qb, mb, dA + dB, i, B.tool)
        for i, leg in enumerate(legs):
            if cut_end[i] - cut_start[i] > MIN_PIECE:
                if leg.is_joint:
                    pieces.append(JointLegPiece(leg, cut_start[i], cut_end[i], i))
                else:
                    pieces.append(CartPieceSmart(LegPiece(leg.geom, cut_start[i], cut_end[i], i), leg.tool, i))
            if i in blends:
                pieces.append(blends[i])
        path = PiecewisePath(pieces)
        if path.length <= 0:
            return None
        grid = self._grid(pieces, float(self.cfg["robot"]["grid_step"]))
        n, nq = len(grid), robot.nq
        q_grid, dq, ddq = np.empty((n, nq)), np.empty((n, nq)), np.empty((n, nq))
        c1, c2 = np.zeros((n, 3)), np.zeros((n, 3))
        vmax, acc_scale, cart_acc = np.empty(n), np.empty(n), np.empty(n)
        seed = legs[0].q0.copy()
        failed_at: set[str] = set()
        L = path.length
        for k, p in enumerate(grid):
            shift = max(0.0, h_fd - p) - max(0.0, p + h_fd - L)
            center = p + shift
            qs, Ts = [], []
            for x in (center - h_fd, center, center + h_fd):
                q, T, ok, src = self._q_at(path, x, seed, legs)
                if not ok:
                    failed_at.add(src)
                seed = q
                qs.append(q)
                Ts.append(T)
            if shift != 0.0:
                q_here, _, ok, src = self._q_at(path, p, qs[1], legs)
            else:
                q_here = qs[1]
            q_grid[k] = q_here
            dq[k] = (qs[2] - qs[0]) / (2 * h_fd)
            ddq[k] = (qs[2] - 2 * qs[1] + qs[0]) / h_fd**2
            pi, local = path.locate(p)
            pc = pieces[pi[0]]
            pos = np.array([T[:3, 3] for T in Ts])
            cart_v = (pos[2] - pos[0]) / (2 * h_fd)
            ori = rotation_angle_deg(Ts[0][:3, :3], Ts[2][:3, :3]) / (2 * h_fd)
            if pc.kind == "cart":
                c1[k] = cart_v
                c2[k] = (pos[2] - 2 * pos[1] + pos[0]) / h_fd**2
            w = self._blend_weight(pc, local[0])
            A = legs[pc.index]
            limA = self._smart_limit(A, dq[k], np.linalg.norm(cart_v), ori)
            if w is None:
                vmax[k], acc_scale[k], cart_acc[k] = limA, A.accel_scale, A.cart_accel
            else:
                B = legs[pc.index + 1]
                limB = self._smart_limit(B, dq[k], np.linalg.norm(cart_v), ori)
                vmax[k] = (1 - w) * limA + w * limB
                acc_scale[k] = (1 - w) * A.accel_scale + w * B.accel_scale
                cart_acc[k] = (1 - w) * A.cart_accel + w * B.cart_accel
            seed = q_here
        for src in sorted(failed_at):
            self.diag.issue("unreachable", src)
            self.diag.warn(f"{src}: IK failed along the path (unreachable or singular); joints are approximate")
        jumps = np.max(np.abs(np.diff(q_grid, axis=0)), axis=1) if n > 1 else np.zeros(0)
        expected = np.max(np.abs(dq[1:]), axis=1) * np.diff(grid) if n > 1 else np.zeros(0)
        for k in np.flatnonzero(jumps > np.maximum(0.1, 10 * expected)):
            src = legs[pieces[path.locate(grid[k])[0][0]].index].source
            self.diag.issue("joint_jump", src)
            self.diag.warn(f"{src}: joint jump of {np.degrees(jumps[k]):.1f} deg along the path "
                           "(singularity or configuration change)", key=f"jump:{src}")
        acc_rows_q1 = dq / acc_scale[:, None]
        acc_rows_q2 = ddq / acc_scale[:, None]
        timed = parameterize(
            grid, vmax, dq, robot.vel_limit,
            np.hstack([acc_rows_q1, c1 / cart_acc[:, None]]),
            np.hstack([acc_rows_q2, c2 / cart_acc[:, None]]),
            np.concatenate([robot.acc_limit, np.ones(3)]),
        )
        outside = np.any((q_grid < robot.lower - 1e-6) | (q_grid > robot.upper + 1e-6), axis=1)
        if outside.any():
            k = int(np.flatnonzero(outside)[0])
            self.diag.issue("joint_limits", legs[pieces[path.locate(grid[k])[0][0]].index].source)
            self.diag.warn(f"{legs[pieces[path.locate(grid[k])[0][0]].index].source}: path exceeds URDF joint limits")
        return _SmartEvaluator(path, timed, run, robot, grid, q_grid, dq, self.output_tool)

    def _leg_q_and_rate(self, leg: Leg, p: float, seed, h):
        if leg.is_joint:
            return leg.q0 + (leg.q1 - leg.q0) * p / leg.plen, (leg.q1 - leg.q0) / leg.plen
        tool_inv = T_inv(leg.tool)
        qs = []
        for x in (p - h, p, p + h):
            T = np.eye(4)
            T[:3, 3] = leg.geom.pos(x)[0]
            T[:3, :3] = leg.geom.rot(x).as_matrix()[0]
            seed, _ = self.robot.ik(T @ tool_inv, seed)
            qs.append(seed)
        return qs[1], (qs[2] - qs[0]) / (2 * h)

    def _q_at(self, path: PiecewisePath, x: float, seed, legs):
        idx, local = path.locate(x)
        pc = path.pieces[idx[0]]
        src = str(legs[pc.index].source)
        if pc.kind == "joint":
            q = pc.q(local[0])[0]
            return q, self.robot.fk(q) @ pc.tool, True, src
        T = np.eye(4)
        T[:3, 3] = pc.piece.pos(local[0])[0]
        T[:3, :3] = pc.piece.rot(local[0]).as_matrix()[0]
        q, ok = self.robot.ik(T @ pc.tool_inv, seed)
        return q, T, ok, src

    def _smart_limit(self, leg: Leg, dq, speed, ori_rate) -> float:
        if leg.is_joint:
            if leg.joint_frac is not None:
                with np.errstate(divide="ignore"):
                    return float(np.min(leg.joint_frac * self.robot.vel_limit / np.maximum(np.abs(dq), 1e-12)))
            return leg.joint_rate if leg.joint_rate is not None else 1e9
        return _cart_limit(leg, speed, ori_rate)

    # -- sampling ---------------------------------------------------------------------
    def _sample(self, timeline, dt) -> TrajectoryResult:
        total = sum(d for d, _, _ in timeline)
        evaluators = [ev for _, ev, _ in timeline if ev is not None]
        nq = self.robot.nq if self.mode == "smart" else 0
        names = self.robot.joint_names if self.mode == "smart" else []
        if not evaluators:
            self.diag.warn("program produced no motion")
            return TrajectoryResult(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3)), None, names, [],
                                    self.diag.messages)
        n = int(math.floor(total / dt + 1e-9)) + 1
        if (n - 1) * dt < total - 1e-9:
            n += 1
        t = np.arange(n) * dt
        pos, rot = np.empty((n, 3)), np.empty((n, 3, 3))
        joints = np.empty((n, nq)) if nq else None
        source = [""] * n
        t0 = 0.0
        last_ev, next_ev = None, iter(evaluators)
        upcoming = next(next_ev)
        for dur, ev, src in timeline:
            t1 = t0 + dur
            is_last = t1 >= total - 1e-12
            sel = np.flatnonzero((t >= t0 - 1e-12) & ((t < t1) | is_last))
            if ev is None:
                ref = last_ev or upcoming
                at_end = last_ev is not None
                if len(sel):
                    P, R, Q, S = ref.pose_at_time(np.array([ref.duration if at_end else 0.0]))
                    pos[sel], rot[sel] = P[0], R[0]
                    if joints is not None:
                        joints[sel] = Q[0]
                    for i in sel:
                        source[i] = str(src)
            else:
                if len(sel):
                    P, R, Q, S = ev.pose_at_time(t[sel] - t0)
                    pos[sel], rot[sel] = P, R
                    if joints is not None:
                        joints[sel] = Q
                    for i, s in zip(sel, S):
                        source[i] = s
                last_ev = ev
                upcoming = next(next_ev, ev)
            t0 = t1
        return TrajectoryResult(t, pos, rot, joints, names, source, self.diag.messages)


def resolve_output_tool(cfg: dict) -> Optional[np.ndarray]:
    spec = cfg.get("output_tool", "active")
    if spec is None or spec == "active":
        return None
    if spec == "flange":
        return np.eye(4)
    if isinstance(spec, (int, str)) and str(spec).isdigit():
        n = int(spec)
        if n == 0:
            return np.eye(4)
        if n not in cfg.get("tools", {}):
            raise ValueError(f"output_tool {n} is not defined under 'tools' in the config")
        return frame_from_spec(cfg["tools"][n])
    return frame_from_spec(spec)


def _cart_limit(leg: Leg, speed: float, ori_rate: float) -> float:
    lim = math.inf
    if speed > 1e-9 and math.isfinite(leg.v_tcp):
        lim = min(lim, leg.v_tcp / speed)
    if ori_rate > 1e-9 and math.isfinite(leg.v_ori):
        lim = min(lim, leg.v_ori / ori_rate)
    return lim if math.isfinite(lim) else 1e9


def _cart_fd(evaluate, grid, h, length):
    """Positions and finite-difference derivatives of a Cartesian path at grid points."""
    shift = np.maximum(0.0, h - grid) - np.maximum(0.0, grid + h - length)
    center = grid + shift
    pm, Rm = evaluate(center - h)
    pc, _ = evaluate(center)
    pp, Rp = evaluate(center + h)
    c1 = (pp - pm) / (2 * h)
    c2 = (pp - 2 * pc + pm) / h**2
    rel = np.einsum("nji,njk->nik", Rm, Rp)
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ori_rate = np.degrees(np.arccos(cos)) / (2 * h)
    p0, _ = evaluate(grid)
    return p0, c1, c2, ori_rate


class _DumbEvaluator:
    def __init__(self, path: PiecewisePath, timed: TimedPath, run: Run, output_tool=None):
        self.path, self.timed, self.run, self.output_tool = path, timed, run, output_tool
        self.duration = timed.duration

    def pose_at_time(self, t):
        p = self.timed.p_at(np.asarray(t, float))
        pos, rot = self.path.evaluate(p)
        idx, _ = self.path.locate(p)
        if self.output_tool is not None:  # re-express the programmed TCP as the requested tool
            for i in np.unique(idx):
                sel = idx == i
                delta = T_inv(self.run.legs[self.path.pieces[i].index].tool) @ self.output_tool
                pos[sel] = pos[sel] + np.einsum("nij,j->ni", rot[sel], delta[:3, 3])
                rot[sel] = rot[sel] @ delta[:3, :3]
        src = [str(self.run.legs[self.path.pieces[i].index].source) for i in idx]
        return pos, rot, None, src


class _SmartEvaluator:
    def __init__(self, path, timed, run, robot, grid, q_grid, dq, output_tool=None):
        self.path, self.timed, self.run, self.robot = path, timed, run, robot
        self.output_tool = output_tool
        self.duration = timed.duration
        self.spline = CubicHermiteSpline(grid, q_grid, dq) if len(grid) > 1 else None
        self.q_end = q_grid[-1]

    def pose_at_time(self, t):
        p = self.timed.p_at(np.asarray(t, float))
        q = self.spline(p) if self.spline is not None else np.repeat(self.q_end[None], len(p), 0)
        idx, _ = self.path.locate(p)
        pos, rot = np.empty((len(p), 3)), np.empty((len(p), 3, 3))
        for k in range(len(p)):
            pc = self.path.pieces[idx[k]]
            T = self.robot.fk(q[k]) @ (pc.tool if self.output_tool is None else self.output_tool)
            pos[k], rot[k] = T[:3, 3], T[:3, :3]
        joints = np.array([self.robot.to_controller(qk) for qk in q])
        src = [str(self.run.legs[self.path.pieces[i].index].source) for i in idx]
        return pos, rot, joints, src


def plan(program: Program, cfg: dict, mode: str = "dumb", dt: float = 0.01, robot=None,
         diag: Optional[Diagnostics] = None) -> TrajectoryResult:
    return Planner(program, cfg, mode, robot, diag).run(dt)
