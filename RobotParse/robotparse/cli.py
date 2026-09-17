"""Command line interface."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_config
from .model import Diagnostics, Dwell, RobotParseError
from .output import DEFAULT_AXIS, OutputOptions, write_trajectory
from .parsers import detect_vendor, parse_program
from .report import EXIT_FAILED, EXIT_STRICT, build_report
from .transforms import xyzabc_from_T


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="robotparse",
        description="Convert KUKA (.src/.dat), FANUC (.ls) or ABB RAPID (.mod/.prg/.pgf) programs into a TCP "
                    "trajectory sampled at a fixed time step (position + tool direction vector).",
        epilog="Exit codes: 0 complete, 1 failed (nothing written), 3 --strict and warnings, "
               "4 output written but incomplete (see the readiness report).",
    )
    ap.add_argument("programs", nargs="+", help="program file(s); the first is the main program")
    ap.add_argument("-o", "--output", help="output .csv or .npz (default: <program>_<mode>.csv)")
    ap.add_argument("-m", "--mode", choices=["dumb", "smart"], default="dumb",
                    help="dumb: Cartesian estimate from speeds and zones; smart: URDF kinematics and joint limits")
    ap.add_argument("--dt", type=_positive_float, default=0.01, help="output time step in seconds (default 0.01)")
    ap.add_argument("-c", "--config", action="append",
                    help="YAML config (frames, limits, joint mapping, registers); repeat to combine files")
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
    ap.add_argument("--no-template", action="store_true", help="don't write the <output>_missing.yaml template")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every warning")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print the result line")
    ap.add_argument("--strict", action="store_true", help="exit with status 3 if there were any warnings")
    ap.add_argument("--debug", action="store_true", help="show Python tracebacks for unexpected errors")
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
    try:
        return run(args)
    except RobotParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - last resort: no tracebacks for users unless asked
        if args.debug:
            raise
        print(f"error: unexpected problem ({type(exc).__name__}: {exc}).\n"
              "This is probably a bug or a program feature robotparse doesn't handle yet; "
              "rerun with --debug to see the details.", file=sys.stderr)
        return EXIT_FAILED


def run(args) -> int:
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
        _print_warnings(diag, args.verbose, args.quiet)
        return 0

    robot = None
    if cfg["robot"].get("urdf"):
        from .kinematics import RobotModel
        robot = RobotModel(cfg["robot"], diag)
    elif args.mode == "smart":
        raise RobotParseError("smart mode needs a robot model: pass --urdf robot.urdf or set robot.urdf in the config")

    from .planner import plan
    t0 = time.time()
    result = plan(program, cfg, args.mode, args.dt, robot, diag)
    elapsed = time.time() - t0

    out = Path(args.output) if args.output else Path(f"{paths[0].stem}_{args.mode}.csv")
    axis = cfg.get("direction_axis") or DEFAULT_AXIS[vendor]
    if len(result.t):
        out.parent.mkdir(parents=True, exist_ok=True)
        opts = OutputOptions(units=cfg["units"], direction_axis=axis, quaternion=args.quat, joints=args.joints,
                             speed=args.speed, source=args.source)
        write_trajectory(result, out, cfg, vendor, opts)
    report = build_report(program, result, cfg, diag, mode=args.mode, dt=args.dt, vendor=vendor,
                          direction_axis=axis, output=str(out), warnings_hidden=not args.verbose)

    template_line = None
    if report.template and not args.no_template:
        tpath = out.with_name(f"{out.stem}_missing.yaml")
        if tpath.exists():
            template_line = f"Template   {tpath} already exists (not overwritten)"
        else:
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.write_text(report.template)
            template_line = f"Template   fill in {tpath}, then run again adding: -c {tpath}"

    if args.quiet:
        print(report.verdict)
    else:
        _print_warnings(diag, args.verbose, quiet=False)
        print(report.text)
        if template_line:
            print(template_line)
        print(f"Time       planned in {elapsed:.1f} s")
    if args.plot and len(result.t):
        from .plot import plot_trajectory
        plot_trajectory(result, args.plot, title=f"{program.name} ({args.mode})", axis=axis)
        if not args.quiet:
            print(f"Plot       {args.plot}")
    if report.exit_code:
        return report.exit_code
    if args.strict and diag.messages:
        print(f"error: --strict and {len(diag.messages)} warning(s)", file=sys.stderr)
        return EXIT_STRICT
    return 0


def _print_warnings(diag: Diagnostics, verbose: bool, quiet: bool) -> None:
    if quiet or not diag.messages or not verbose:
        return
    print(f"{len(diag.messages)} warning(s):", file=sys.stderr)
    for msg in diag.messages:
        print(f"  - {msg}", file=sys.stderr)
