"""Pose helpers and vendor orientation conventions.

All positions are millimetres and all angles are degrees unless a name says otherwise.
Poses are 4x4 homogeneous numpy arrays.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(position, rotation) -> np.ndarray:
    """Build a 4x4 pose from a position and a scipy Rotation or 3x3 matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix() if isinstance(rotation, Rotation) else np.asarray(rotation)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def translation(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return make_T([x, y, z], np.eye(3))


def T_inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def T_from_xyzabc(x, y, z, a, b, c) -> np.ndarray:
    """KUKA convention: A about Z, then B about the new Y, then C about the new X."""
    return make_T([x, y, z], Rotation.from_euler("ZYX", [a, b, c], degrees=True))


def xyzabc_from_T(T: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # gimbal lock at B = +-90 is expected and harmless here
        a, b, c = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)
    return np.array([*T[:3, 3], a, b, c])


def T_from_xyzwpr(x, y, z, w, p, r) -> np.ndarray:
    """FANUC convention: W about fixed X, P about fixed Y, R about fixed Z (Rz*Ry*Rx)."""
    return make_T([x, y, z], Rotation.from_euler("xyz", [w, p, r], degrees=True))


def xyzwpr_from_T(T: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, p, r = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return np.array([*T[:3, 3], w, p, r])


def T_from_pos_quat(pos, q_wxyz) -> np.ndarray:
    """ABB convention: quaternion given scalar-first as [q1, q2, q3, q4] = [w, x, y, z]."""
    w, x, y, z = (float(v) for v in q_wxyz)
    return make_T(pos, Rotation.from_quat([x, y, z, w]))


def quat_wxyz_from_T(T: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z])


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_axis(spec: str) -> tuple[int, float]:
    """'z', '+x', '-y' -> (column index, sign)."""
    s = spec.strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    if s not in AXIS_INDEX:
        raise ValueError(f"direction axis must be one of x, y, z (optionally signed), got {spec!r}")
    return AXIS_INDEX[s], sign


def frame_from_spec(spec) -> np.ndarray:
    """Build a pose from a config entry.

    Accepted forms:
      * 4x4 nested list
      * {xyz: [x, y, z], abc: [a, b, c]}   (KUKA ZYX Euler)
      * {xyz: [...], wpr: [w, p, r]}        (FANUC fixed XYZ; same as ROS rpy)
      * {xyz: [...], rpy: [r, p, y]}        (degrees)
      * {xyz: [...], quat: [w, x, y, z]}    (ABB order)
    """
    if spec is None:
        return np.eye(4)
    if isinstance(spec, np.ndarray):
        return spec.astype(float)
    if isinstance(spec, (list, tuple)):
        arr = np.asarray(spec, dtype=float)
        if arr.shape == (4, 4):
            return arr
        raise ValueError(f"frame list must be a 4x4 matrix, got shape {arr.shape}")
    xyz = spec.get("xyz", [0.0, 0.0, 0.0])
    if "abc" in spec:
        return T_from_xyzabc(*xyz, *spec["abc"])
    if "wpr" in spec:
        return T_from_xyzwpr(*xyz, *spec["wpr"])
    if "rpy" in spec:
        return T_from_xyzwpr(*xyz, *spec["rpy"])
    if "quat" in spec:
        return T_from_pos_quat(xyz, spec["quat"])
    if "matrix" in spec:
        return np.asarray(spec["matrix"], dtype=float)
    return translation(*xyz)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of the relative rotation between two 3x3 matrices."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
