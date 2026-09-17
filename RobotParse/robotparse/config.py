"""Configuration: defaults plus an optional YAML file (see examples/*.yaml)."""
from __future__ import annotations

import copy
import difflib
import math
import re
from pathlib import Path

import numpy as np
import yaml

from .model import RobotParseError
from .transforms import frame_from_spec, parse_axis

DEFAULTS: dict = {
    # Tool axis reported as the "direction" vector. None -> vendor default
    # (KUKA: x, the KUKA tool convention; ABB and FANUC: z).
    "direction_axis": None,
    "units": "mm",  # output length units: mm or m
    # Pose of the robot base in the output/world frame.
    "world_from_robot": None,
    # Tool / base (user frame) tables by number: KUKA TOOL_DATA[n]/BASE_DATA[n],
    # FANUC UTOOL n / UFRAME n. Each entry is a frame spec (see transforms.frame_from_spec).
    "tools": {},
    "bases": {},
    "entry": None,  # routine to execute; default main/first routine
    # Which TCP to report: "active" (the tool active in the program), "flange", a tool number
    # from `tools`, or a frame spec. A fixed tool avoids jumps when the program switches tools
    # (e.g. KUKA PTP HOME inline forms that use tool 0).
    "output_tool": "active",
    "motion": {
        "cart_accel": 2500.0,  # mm/s^2 path acceleration limit for Cartesian moves
        "default_tcp_speed": 250.0,  # mm/s when a program never sets one
        "default_ori_speed": 200.0,  # deg/s when a program never sets one
        "max_ori_speed": 500.0,  # deg/s cap for orientation (FANUC, KUKA $VEL.ORI)
        "ptp_tcp_speed": 2000.0,  # dumb mode: TCP speed of a 100% joint move
        "ptp_ori_speed": 360.0,  # dumb mode: orientation speed of a 100% joint move
        "fanuc_cnt_time": 0.25,  # s; CNT100 rounds corners by about speed * this time
        "kuka_spline_join": 0.25,  # fraction of shorter segment blended inside SPLINE blocks
        "grid_step": 2.0,  # mm between time-parameterisation grid points (dumb mode)
    },
    "robot": {
        "urdf": None,
        "tip_link": None,  # URDF link of the controller flange (default: tool0/flange/last link)
        "tip_offset": None,  # frame spec: controller flange relative to tip_link
        "joint_offsets": None,  # deg: controller = sign * urdf + offset
        "joint_signs": None,
        "fanuc_j23_coupling": False,  # FANUC J3 is measured from horizontal: urdf_j3 = j3 + j2
        "joint_vel_limits": None,  # deg/s, overrides URDF <limit velocity>
        "joint_acc_limits": None,  # deg/s^2
        "accel_time": 0.4,  # s to reach full joint speed when acc limits are not given
        "initial_joints": None,  # controller deg; start state for smart mode
        "grid_step": 4.0,  # mm between IK/time-parameterisation grid points (smart mode)
        "ik_random_seeds": 24,
    },
    "fanuc": {
        "registers": {},  # R[n]: value
        "position_registers": {},  # PR[n]: {xyz, wpr} or {joints}
    },
    "kuka": {
        "include_dat": [],  # extra .dat files, e.g. $config.dat for TOOL_DATA/BASE_DATA
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


SECTIONS = ("motion", "robot", "fanuc", "kuka")
ROTATION_KEYS = ("abc", "wpr", "rpy", "quat", "matrix")


def load_config(paths=None, overrides: dict | None = None) -> dict:
    """Defaults, then each YAML file in order (later files override earlier ones), then overrides."""
    if paths is None:
        paths = []
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    cfg = copy.deepcopy(DEFAULTS)
    user_set: set[str] = set()
    for path in paths:
        path = Path(path)
        data = _read_yaml(path)
        _check_keys(data, path.name)
        _check_todo(data, path.name)
        base_dir = path.resolve().parent  # relative file paths are relative to the config file
        robot = data.get("robot") or {}
        if robot.get("urdf") and not Path(str(robot["urdf"])).is_absolute():
            robot["urdf"] = str(base_dir / str(robot["urdf"]))
        kuka = data.get("kuka") or {}
        if kuka.get("include_dat"):
            kuka["include_dat"] = [str(p) if Path(str(p)).is_absolute() else str(base_dir / str(p))
                                   for p in kuka["include_dat"]]
        user_set |= _dotted(data)
        cfg = _merge(cfg, data)
    if overrides:
        user_set |= _dotted(overrides)
        cfg = _merge(cfg, overrides)
    _normalise_and_validate(cfg)
    cfg["_user_set"] = user_set
    return cfg


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise RobotParseError(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise RobotParseError(f"{path.name}: invalid YAML{where}: {problem}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RobotParseError(f"{path.name}: expected 'key: value' settings at the top level")
    return data


def _unknown_key(key: str, allowed, where: str) -> RobotParseError:
    guess = difflib.get_close_matches(str(key), [str(a) for a in allowed], n=1)
    hint = f" Did you mean '{guess[0]}'?" if guess else ""
    return RobotParseError(f"{where}: unknown setting '{key}'.{hint}")


def _check_keys(data: dict, name: str) -> None:
    for key, value in data.items():
        if key not in DEFAULTS or str(key).startswith("_"):
            raise _unknown_key(key, list(DEFAULTS), name)
        if key in SECTIONS:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise RobotParseError(f"{name}: '{key}' must contain settings (key: value lines)")
            for sub in value:
                if sub not in DEFAULTS[key]:
                    raise _unknown_key(f"{key}.{sub}", [f"{key}.{k}" for k in DEFAULTS[key]], name)


def _check_todo(data, name: str) -> None:
    todo: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and node.strip().upper().startswith("TODO"):
            todo.append(path)

    walk(data, "")
    entries = []
    for path in todo:  # report entries (tools.1, robot.urdf), not every list element
        path = re.sub(r"\[\d+\]$", "", path)
        m = re.match(r"^(tools\.[^.]+|bases\.[^.]+|fanuc\.\w+\.[^.]+)", path)
        entries.append(m.group(1) if m else path)
    todo = list(dict.fromkeys(entries))
    if todo:
        shown = ", ".join(todo[:8]) + (f" and {len(todo) - 8} more" if len(todo) > 8 else "")
        raise RobotParseError(f"{name}: fill in the TODO placeholders, or delete the entries you don't have: {shown}")


def _dotted(data, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if v is not None and v != {}:
                out.add(key)
            out |= _dotted(v, key)
    return out


def _numbers(value, n, where: str) -> None:
    if not isinstance(value, (list, tuple)) or (n and len(value) != n) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        size = f"{n} " if n else ""
        raise RobotParseError(f"{where}: expected a list of {size}numbers, got {value!r}")


def validate_frame(spec, where: str) -> None:
    if isinstance(spec, dict):
        unknown = set(spec) - {"xyz", *ROTATION_KEYS}
        if unknown:
            raise RobotParseError(f"{where}: unknown frame keys {sorted(unknown)}; use xyz plus one of "
                                  "abc (KUKA), wpr (FANUC), rpy or quat [w, x, y, z]")
        rots = [k for k in ROTATION_KEYS if k in spec]
        if len(rots) > 1:
            raise RobotParseError(f"{where}: give only one of {', '.join(rots)}")
        if "xyz" in spec:
            _numbers(spec["xyz"], 3, f"{where}.xyz")
        for k in rots:
            if k == "matrix":
                continue
            _numbers(spec[k], 4 if k == "quat" else 3, f"{where}.{k}")
        if "quat" in spec and not any(spec["quat"]):
            raise RobotParseError(f"{where}.quat: quaternion must not be all zeros")
    try:
        T = frame_from_spec(spec)
    except (ValueError, TypeError, KeyError) as exc:
        raise RobotParseError(f"{where}: {exc}. Expected e.g. {{xyz: [x, y, z], abc: [a, b, c]}} "
                              "(mm and degrees)") from None
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise RobotParseError(f"{where}: invalid frame")


def _normalise_and_validate(cfg: dict) -> None:
    for table in ("tools", "bases"):
        entries = cfg.get(table) or {}
        if not isinstance(entries, dict):
            raise RobotParseError(f"'{table}' must map numbers to frames, e.g. '{table}: {{1: {{xyz: [...], abc: [...]}}}}'")
        out = {}
        for k, v in entries.items():
            try:
                n = int(k)
            except (TypeError, ValueError):
                raise RobotParseError(f"{table}: '{k}' is not a tool/frame number") from None
            validate_frame(v, f"{table}.{n}")
            out[n] = v
        cfg[table] = out
    fanuc = cfg["fanuc"]
    for table in ("registers", "position_registers"):
        entries = fanuc.get(table) or {}
        out = {}
        for k, v in entries.items():
            try:
                n = int(k)
            except (TypeError, ValueError):
                raise RobotParseError(f"fanuc.{table}: '{k}' is not a register number") from None
            where = f"fanuc.{table}.{n}"
            if table == "registers":
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise RobotParseError(f"{where}: expected a number, got {v!r}")
            elif isinstance(v, dict) and "joints" in v:
                _numbers(v["joints"], 0, f"{where}.joints")
            else:
                validate_frame(v, where)
            out[n] = v
        fanuc[table] = out
    for key in ("world_from_robot",):
        if cfg.get(key) is not None:
            validate_frame(cfg[key], key)
    if cfg["robot"].get("tip_offset") is not None:
        validate_frame(cfg["robot"]["tip_offset"], "robot.tip_offset")
    tool = cfg.get("output_tool", "active")
    if not (tool is None or tool in ("active", "flange") or isinstance(tool, int)
            or (isinstance(tool, str) and tool.isdigit())):
        validate_frame(tool, "output_tool")
    if cfg.get("units") not in ("mm", "m"):
        raise RobotParseError(f"units must be 'mm' or 'm', got {cfg.get('units')!r}")
    if cfg.get("direction_axis") is not None:
        try:
            parse_axis(str(cfg["direction_axis"]))
        except ValueError as exc:
            raise RobotParseError(str(exc)) from None
    for key, value in cfg["motion"].items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise RobotParseError(f"motion.{key} must be a positive number, got {value!r}")
    for key in ("joint_offsets", "joint_signs", "joint_vel_limits", "joint_acc_limits", "initial_joints"):
        if cfg["robot"].get(key) is not None:
            _numbers(cfg["robot"][key], 0, f"robot.{key}")
    if isinstance(cfg["kuka"].get("include_dat"), (str, Path)):
        cfg["kuka"]["include_dat"] = [cfg["kuka"]["include_dat"]]
