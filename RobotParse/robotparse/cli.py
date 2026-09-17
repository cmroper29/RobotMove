"""Command line interface."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


from .config import load_config
from .model import Diagnostics, Dwell
from .output import DEFAULT_AXIS, OutputOptions, write_trajectory
from .parsers import detect_vendor, parse_program
from .transforms import xyzabc_from_T


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="robotparse",
        description="Convert KUKA (.src/.dat), FANUC (.ls) or ABB RAPID (.mod/.prg) programs into a TCP "
                    "trajectory sampled at a fixed time step (position + tool direction vector).",
    )
    ap.add_argument("programs", nargs="+", help="program file(s); the first is the main program")
    ap.add_argument("-o", "--output", help="output .csv or .npz (default: <program>_<mode>.csv)")
    ap.add_argument("-m", "--mode", choices=["dumb", "smart"], default="dumb",
                    help="dumb: Cartesian estimate from speeds and zones; smart: URDF kinematics and joint limits")
    ap.add_argument("--dt", type=float, default=0.01, help="output time step in seconds (default 0.01)")
    ap.add_argument("-c", "--config", help="YAML config (frames, limits, joint mapping, registers)")
    ap.add_argument("--urdf", help="robot URDF (required for smart mode, optional in dumb mode for joint targets)")
    ap.add_argument("--tip-link", help="URDF link that corresponds to the controller flange")
    ap.add_argument("--vendor", choices=["kuka", "fanuc", "abb"], help="override language detection")
    ap.add_argument("--entry", help="routine/program to start from (default: main / file name)")
    ap.add_argument("--units", choices=["mm", "m"], help="output length units (default mm)")
    ap.add_argument("--axis", help="tool axis used as the direction vector, e.g. z, x, -z (default: KUKA x, ABB/FANUC z)")
    ap.add_argument("--quat", action="store_true", help="also write the TCP orientation quaternion (qw,qx,qy,qz)")
    ap.add_argument("--joints", action="store_true", help="also write joint values (smart mode)")
    ap.add_argument("--speed", action="store_true", help="also write TCP speed")
    ap.add_argument("--source", action="store_true", help="also write the program line being executed")
    ap.add_argument("--plot", help="save an overview plot (PNG) of the trajectory; needs matplotlib")
    ap.add_argument("--list", action="store_true", help="print the parsed motion list and exit")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress warnings")
    ap.add_argument("--strict", action="store_true", help="exit with status 3 if there were any warnings")
    return ap


def print_program(program) -> None:
    print(f"{program.vendor.upper()} program '{program.name}': {len(program.motions)} motions")
    for c in program.commands:
        if isinstance(c, Dwell):
            print(f"  {str(c.source):>18}  WAIT {c.duration:.3f}s")
            continue
        tgt = c.target
        if hasattr(tgt, "T"):
            desc = "xyzabc=" + ",".join(f"{v:.1f}" for v in xyzabc_from_T(c.base @ tgt.T))
        elif hasattr(tgt, "q"):
            desc = "joints=" + ",".join(f"{v:.1f}" for v in tgt.q)
        else:
            desc = "relative"
        sp = c.speed
        speed = (f"{sp.joint_pct:.0f}%" if sp.joint_pct is not None else
                 f"{sp.duration:.2f}s" if sp.duration else f"{sp.tcp or 0:.0f}mm/s")
        z = c.zone
        zone = "fine" if z.fine else (f"d={z.dist:.1f}" if z.dist is not None else
                                      f"cnt={z.cnt:.0f}" if z.cnt is not None else f"pct={z.pct:.0f}")
        print(f"  {str(c.source):>18}  {c.kind:<6} {desc:<52} {speed:>10} {zone}")


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    overrides: dict = {"robot": {}}
    if args.urdf:
        overrides["robot"]["urdf"] = str(Path(args.urdf).resolve())
    if args.tip_link:
        overrides["robot"]["tip_link"] = args.tip_link
    if args.entry:
        overrides["entry"] = args.entry
    if args.units:
        overrides["units"] = args.units
    if args.axis:
        overrides["direction_axis"] = args.axis
    cfg = load_config(args.config, overrides)

    paths = [Path(p) for p in args.programs]
    vendor = args.vendor or detect_vendor(paths[0])
    diag = Diagnostics()
    program = parse_program(paths, cfg, vendor, diag)
    if args.list:
        print_program(program)
        _print_warnings(diag, args.quiet)
        return 0

    robot = None
    if cfg["robot"].get("urdf"):
        from .kinematics import RobotModel
        robot = RobotModel(cfg["robot"], diag)
    elif args.mode == "smart":
        print("error: smart mode needs --urdf or robot.urdf in the config", file=sys.stderr)
        return 2

    from .planner import PlanningError, plan
    t0 = time.time()
    try:
        result = plan(program, cfg, args.mode, args.dt, robot, diag)
    except PlanningError as exc:
        _print_warnings(diag, args.quiet)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    out = Path(args.output) if args.output else Path(f"{paths[0].stem}_{args.mode}.csv")
    opts = OutputOptions(
        units=cfg["units"],
        direction_axis=cfg.get("direction_axis") or DEFAULT_AXIS[vendor],
        quaternion=args.quat, joints=args.joints, speed=args.speed, source=args.source,
    )
    write_trajectory(result, out, cfg, vendor, opts)
    _print_warnings(diag, args.quiet)
    dur = result.t[-1] if len(result.t) else 0.0
    print(f"{vendor.upper()} '{program.name}': {len(program.motions)} motions -> {len(result.t)} samples, "
          f"{dur:.3f} s at dt={args.dt} ({args.mode} mode, {elapsed:.1f} s) -> {out}")
    if args.plot:
        from .plot import plot_trajectory
        plot_trajectory(result, args.plot, title=f"{program.name} ({args.mode})", axis=opts.direction_axis)
        print(f"plot -> {args.plot}")
    if args.strict and diag.messages:
        print(f"error: --strict and {len(diag.messages)} warning(s)", file=sys.stderr)
        return 3
    return 0


def _print_warnings(diag: Diagnostics, quiet: bool) -> None:
    if quiet or not diag.messages:
        return
    print(f"{len(diag.messages)} warning(s):", file=sys.stderr)
    for msg in diag.messages:
        print(f"  - {msg}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
