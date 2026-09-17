"""Writing trajectories to CSV / NPZ."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .planner import TrajectoryResult
from .transforms import frame_from_spec, parse_axis

DEFAULT_AXIS = {"kuka": "x", "abb": "z", "fanuc": "z"}


@dataclass
class OutputOptions:
    units: str = "mm"
    direction_axis: str = "z"
    quaternion: bool = False
    joints: bool = False
    speed: bool = False
    source: bool = False


def to_table(result: TrajectoryResult, cfg: dict, vendor: str, opts: OutputOptions) -> tuple[list[str], np.ndarray, list]:
    world = frame_from_spec(cfg.get("world_from_robot"))
    Rw, pw = world[:3, :3], world[:3, 3]
    pos = result.position @ Rw.T + pw
    rot = Rw @ result.rotation
    idx, sign = parse_axis(opts.direction_axis)
    direction = sign * rot[:, :, idx]
    scale = {"mm": 1.0, "m": 0.001}[opts.units]
    u = opts.units
    cols = ["t", f"x_{u}", f"y_{u}", f"z_{u}", "dir_x", "dir_y", "dir_z"]
    data = [result.t[:, None], pos * scale, direction]
    if opts.quaternion:
        xyzw = Rotation.from_matrix(rot).as_quat() if len(rot) else np.zeros((0, 4))
        cols += ["qw", "qx", "qy", "qz"]
        data.append(np.column_stack([xyzw[:, 3], xyzw[:, :3]]))
    if opts.speed:
        if len(pos) > 1:
            v = np.linalg.norm(np.gradient(pos * scale, result.t, axis=0), axis=1)
        else:
            v = np.zeros(len(pos))
        cols.append(f"speed_{u}_s")
        data.append(v[:, None])
    if opts.joints and result.joints is not None:
        cols += [f"{name}" for name in result.joint_names]
        data.append(result.joints)
    table = np.hstack(data) if len(result.t) else np.zeros((0, len(cols)))
    extra = result.source if opts.source else []
    if opts.source:
        cols.append("source")
    return cols, table, extra


def write_trajectory(result: TrajectoryResult, path: str | Path, cfg: dict, vendor: str, opts: OutputOptions) -> Path:
    path = Path(path)
    cols, table, extra = to_table(result, cfg, vendor, opts)
    if path.suffix.lower() == ".npz":
        payload = {c: table[:, i] for i, c in enumerate(cols) if c != "source"}
        if extra:
            payload["source"] = np.array(extra)
        np.savez_compressed(path, **payload)
        return path
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, row in enumerate(table):
            vals = [f"{row[0]:.6f}"] + [f"{v:.6f}" for v in row[1:]]
            if extra:
                vals.append(extra[i])
            w.writerow(vals)
    return path
