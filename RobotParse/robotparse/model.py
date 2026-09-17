"""Vendor-neutral representation of a parsed robot program."""
from __future__ import annotations

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

    @property
    def motions(self) -> list[Motion]:
        return [c for c in self.commands if isinstance(c, Motion)]


class Diagnostics:
    """Collects de-duplicated warnings while parsing and planning."""

    def __init__(self):
        self.messages: list[str] = []
        self._seen: set[str] = set()

    def warn(self, msg: str, key: Optional[str] = None) -> None:
        k = key or msg
        if k in self._seen:
            return
        self._seen.add(k)
        self.messages.append(msg)
