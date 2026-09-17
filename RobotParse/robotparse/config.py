"""Configuration: defaults plus an optional YAML file (see examples/*.yaml)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

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


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        cfg = _merge(cfg, data)
        base_dir = Path(path).resolve().parent
        # resolve relative file paths against the config file location
        if cfg["robot"].get("urdf") and not Path(cfg["robot"]["urdf"]).is_absolute():
            cfg["robot"]["urdf"] = str(base_dir / cfg["robot"]["urdf"])
        cfg["kuka"]["include_dat"] = [
            p if Path(p).is_absolute() else str(base_dir / p) for p in cfg["kuka"].get("include_dat", [])
        ]
    if overrides:
        cfg = _merge(cfg, overrides)
    # YAML may give integer keys or strings; normalise frame tables to int keys
    for table in ("tools", "bases"):
        cfg[table] = {int(k): v for k, v in (cfg.get(table) or {}).items()}
    for table in ("registers", "position_registers"):
        cfg["fanuc"][table] = {int(k): v for k, v in (cfg["fanuc"].get(table) or {}).items()}
    return cfg
