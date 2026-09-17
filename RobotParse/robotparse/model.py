"""Vendor-neutral representation of a parsed robot program."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class Source:
    file: str
    line: int
    text: str = ""

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class CartTarget:
    """TCP pose expressed in the base/user frame that was active when it was programmed."""

    T: np.ndarray
    # Normalised configuration hints used to pick an IK solution, e.g.
    # {"kuka_T": 42}, {"abb_cf": [0, -1, 0, 1]}, {"fanuc_turns": [0, 0, 0]}
    config: dict = field(default_factory=dict)


@dataclass
class JointTarget:
    """Joint values in controller units (degrees for rotary axes)."""

    q: np.ndarray


@dataclass
class RelativeTarget:
    """Target defined relative to wherever the robot currently is."""

    delta: np.ndarray = field(default_factory=lambda: np.eye(4))
    frame: str = "base"  # "base": translate/rotate in base axes, "tool": post-multiply
    joints: Optional[np.ndarray] = None  # joint increments (deg) instead of a Cartesian delta


Target = Union[CartTarget, JointTarget, RelativeTarget]


@dataclass
class Speed:
    tcp: Optional[float] = None  # mm/s
    ori: Optional[float] = None  # deg/s
    joint_pct: Optional[float] = None  # % of maximum joint speed (joint moves)
    duration: Optional[float] = None  # s, for time-programmed moves
    accel_pct: float = 100.0  # scales joint and Cartesian acceleration limits
    cart_accel: Optional[float] = None  # mm/s^2, absolute Cartesian accel (e.g. KUKA $ACC.CP)


@dataclass
class Zone:
    """How the motion ends: exact stop (fine) or blended into the next motion."""

    fine: bool = True
    dist: Optional[float] = None  # blend distance from the corner, mm
    pct: Optional[float] = None  # blend size as % of half the shorter adjacent segment
    cnt: Optional[float] = None  # FANUC CNT value 0..100 (speed dependent rounding)
    ori: Optional[float] = None  # orientation zone, deg (informational)

    @classmethod
    def blended(cls, **kw) -> "Zone":
        return cls(fine=False, **kw)


@dataclass
class Motion:
    kind: str  # "PTP" | "LIN" | "CIRC" | "SPLINE"
    target: Target
    speed: Speed
    zone: Zone
    tool: np.ndarray  # TCP pose in the flange frame
    base: np.ndarray  # active base/user frame in robot base coordinates
    source: Source
    via: Optional[Target] = None  # CIRC: point on the arc between start and end
    arc_ref_excluded: bool = False  # True if `via` lies on the circle but NOT between start and end
    circ_angle: Optional[float] = None  # KUKA CA: total arc angle in degrees


@dataclass
class Dwell:
    duration: float
    source: Source


Command = Union[Motion, Dwell]


@dataclass
class Program:
    vendor: str
    name: str
    commands: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    diag: Optional["Diagnostics"] = None

    @property
    def motions(self) -> list[Motion]:
        return [c for c in self.commands if isinstance(c, Motion)]


class RobotParseError(Exception):
    """A problem with the user's files or settings, reported without a traceback."""


@dataclass
class Need:
    """Controller data or a file the program needs but that was not provided."""

    kind: str  # tool, base, position_register, register, urdf, name, file, routine
    key: str
    where: str = ""
    count: int = 1


class Diagnostics:
    """Collects warnings plus structured facts for the readiness report."""

    def __init__(self):
        self.messages: list[str] = []
        self._seen: set[str] = set()
        self.motion_statements = 0  # motion instructions executed by the parser (loops count repeatedly)
        self.dropped: list[tuple[str, str]] = []  # (where, reason) for motions that could not be used
        self.needs: dict[tuple[str, str], Need] = {}
        self.issues: dict[str, list[str]] = {}  # category -> places (e.g. unreachable targets)
        self.counts: Counter = Counter()  # assumption counters (runtime conditions, tool changes, ...)

    def warn(self, msg: str, key: Optional[str] = None) -> bool:
        k = key or msg
        if k in self._seen:
            return False
        self._seen.add(k)
        self.messages.append(msg)
        return True

    def drop(self, where, reason: str) -> None:
        self.dropped.append((str(where), reason))
        self.warn(f"{where}: {reason}; motion skipped")

    def need(self, kind: str, key, where="") -> None:
        k = (kind, str(key))
        if k in self.needs:
            self.needs[k].count += 1
        else:
            self.needs[k] = Need(kind, str(key), str(where))

    def issue(self, category: str, where) -> None:
        places = self.issues.setdefault(category, [])
        if str(where) not in places:
            places.append(str(where))
