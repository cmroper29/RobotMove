"""Readiness report: what was used, what is missing, what was assumed, plus a config template."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import Diagnostics, Program

EXIT_COMPLETE, EXIT_FAILED, EXIT_STRICT, EXIT_INCOMPLETE = 0, 1, 3, 4
BLOCKING_ISSUES = ("unreachable", "joint_limits", "joint_jump")

AXIS_CONVENTION = {"kuka": "KUKA tool convention", "abb": "ABB tool convention", "fanuc": "FANUC tool convention"}


@dataclass
class ReadinessReport:
    status: str  # COMPLETE, INCOMPLETE or FAILED
    exit_code: int
    text: str
    template: Optional[str]  # YAML with TODO placeholders for missing controller data
    verdict: str = ""


def _places(places: list[str], limit: int = 4) -> str:
    places = list(dict.fromkeys(places))
    shown = ", ".join(places[:limit])
    return shown + (f" and {len(places) - limit} more" if len(places) > limit else "")


def _wrap(prefix: str, text: str, width: int = 100) -> list[str]:
    words, lines, line = text.split(), [], prefix
    indent = " " * len(prefix)
    for w in words:
        if len(line) + len(w) + 1 > width and line.strip():
            lines.append(line.rstrip())
            line = indent
        line += w + " "
    lines.append(line.rstrip())
    return lines


def _tool_label(vendor: str, kind: str, key: str) -> str:
    if vendor == "kuka":
        return f"{'TOOL_DATA' if kind == 'tool' else 'BASE_DATA'}[{key}]"
    return f"{'UTOOL' if kind == 'tool' else 'UFRAME'} {key}"


def _missing_lines(diag: Diagnostics, vendor: str) -> list[str]:
    out: list[str] = []
    needs = list(diag.needs.values())
    by_kind: dict[str, list] = {}
    for n in needs:
        by_kind.setdefault(n.kind, []).append(n)
    for n in by_kind.get("file", []):
        out += _wrap("  - ", f"File {n.key} was not provided: {n.where}. Pass it together with the program.")
    for n in by_kind.get("urdf", []):
        out += _wrap("  - ", f"Robot model (URDF): {n.count} joint-position move(s) need one to compute the TCP "
                             f"(first at {n.where}). Pass --urdf or set robot.urdf.")
    for kind in ("tool", "base"):
        for n in by_kind.get(kind, []):
            label = _tool_label(vendor, kind, n.key)
            if vendor == "kuka":
                where_to_find = f"{label} in KRC:\\R1\\System\\$config.dat"
            else:
                where_to_find = f"MENU > SETUP > Frames > {'Tool' if kind == 'tool' else 'User'} Frame on the pendant"
            out += _wrap("  - ", f"{label} (first used at {n.where or 'the program'}) is unknown; identity was used. "
                                 f"Find it at {where_to_find}.")
    for n in by_kind.get("position_register", []):
        out += _wrap("  - ", f"PR[{n.key}] (first used at {n.where}) has no value. Find it under DATA > Position Reg "
                             "on the pendant (POSREG.VA in a backup).")
    for n in by_kind.get("register", []):
        out += _wrap("  - ", f"R[{n.key}] (first used at {n.where}) has no value; 0 was used. Find it under "
                             "DATA > Registers (NUMREG.VA in a backup).")
    names = by_kind.get("name", [])
    if names:
        where = {"kuka": "the program's .dat, other modules' .dat files, or $config.dat via kuka.include_dat",
                 "abb": "the other modules of the task (.mod/.sys), e.g. tool and work-object data"}.get(vendor, "")
        out += _wrap("  - ", f"Not defined in the loaded files: {_places([n.key for n in names], 8)}. "
                             f"Pass the files that declare them{': ' + where if where else ''}.")
    routines = by_kind.get("routine", [])
    if routines:
        out += _wrap("  - ", f"Routines/programs not loaded: {_places([n.key for n in routines], 8)}. "
                             "Pass the files that define them.")
    return out


def _problem_lines(diag: Diagnostics) -> list[str]:
    texts = {
        "unreachable": "Unreachable for this robot model (IK failed) at {}. Check the tool and base frames, joint "
                       "offsets, and that the URDF is the right robot.",
        "joint_limits": "Path exceeds the URDF joint limits at {}.",
        "joint_jump": "Large joint jump (singularity or arm configuration change) at {}.",
        "config_mismatch": "No joint solution matches the programmed arm configuration at {}; the closest was used.",
    }
    out: list[str] = []
    for cat, text in texts.items():
        if diag.issues.get(cat):
            out += _wrap("  - ", text.format(_places(diag.issues[cat])))
    return out


def _assumption_lines(diag: Diagnostics, cfg: dict, vendor: str, mode: str, axis: str) -> list[str]:
    user = cfg.get("_user_set", set())
    c = diag.counts
    out: list[str] = []
    source = "from the config" if "direction_axis" in user else AXIS_CONVENTION.get(vendor, "default")
    out += _wrap("  - ", f"Direction vector = tool {axis} axis ({source}). Set direction_axis if your tools point "
                         "along another axis.")
    if "world_from_robot" not in user:
        out += _wrap("  - ", "Positions are in the robot base frame; set world_from_robot to place the robot in "
                             "your simulation.")
    if mode == "dumb" and c["dumb_joint_moves"]:
        out += _wrap("  - ", f"{c['dumb_joint_moves']} joint move(s) are drawn as straight lines; a real robot's joint "
                             "moves curve. Use --mode smart with a URDF for their true path.")
    if mode == "dumb":
        accel = ("from the config" if "motion.cart_accel" in user
                 else f"an estimate: motion.cart_accel = {cfg['motion']['cart_accel']:g} mm/s^2")
    else:
        accel = ("from robot.joint_acc_limits" if "robot.joint_acc_limits" in user else
                 f"an estimate (full speed reached in robot.accel_time = {cfg['robot']['accel_time']:g} s); "
                 "set robot.joint_acc_limits from the robot datasheet")
    out += _wrap("  - ", f"Speeds assume 100 % override. Acceleration is {accel}.")
    if c["runtime_condition"]:
        out += _wrap("  - ", f"{c['runtime_condition']} condition(s) or loop(s) depend on run-time state (inputs, "
                             "values set on the controller); the branch taken is listed in the warnings.")
    if c["tool_change"] and cfg.get("output_tool", "active") in (None, "active"):
        out += _wrap("  - ", "The program switches tools, so the reported TCP jumps where it does; set output_tool "
                             "to report one fixed TCP.")
    if c["kuka_default_home"]:
        out += _wrap("  - ", "HOME (XHOME) was not loaded, so the factory default A1..A6 = 0, -90, 90, 0, 0, 0 was "
                             "assumed; pass $config.dat via kuka.include_dat.")
    ignored = diag.issues.get("ignored_instruction")
    if ignored:
        out += _wrap("  - ", f"Instructions ignored: {_places(ignored, 8)}. If any are your own routines, pass the "
                             "module that defines them.")
    return out


def build_report(program: Program, result, cfg: dict, diag: Diagnostics, *, mode: str, dt: float, vendor: str,
                 direction_axis: str, output: Optional[str] = None, warnings_hidden: bool = True) -> ReadinessReport:
    rule = "=" * 72
    lines = [rule, f"Readiness report: {program.name} ({vendor.upper()}, {mode} mode)", rule]
    total = diag.motion_statements
    skipped = len(diag.dropped)
    lines.append(f"Motions    {total - skipped} of {total} used" + (f", {skipped} skipped" if skipped else ""))
    has_output = result is not None and len(result.t) > 0
    if has_output:
        lines.append(f"Output     {len(result.t)} samples every {dt:g} s ({result.t[-1]:.2f} s)"
                     + (f" -> {output}" if output else ""))
    else:
        lines.append("Output     none: no motion could be computed")

    if diag.dropped:
        lines += ["", "Skipped motions"]
        groups: dict[str, list[str]] = {}
        for where, reason in diag.dropped:
            groups.setdefault(reason, []).append(where)
        for reason, places in groups.items():
            lines += _wrap("  - ", f"{reason}: {_places(places)}")
    missing = _missing_lines(diag, vendor)
    if missing:
        lines += ["", "Missing information (defaults were used, so the result may be wrong)"] + missing
    problems = _problem_lines(diag)
    if problems:
        lines += ["", "Problems"] + problems
    lines += ["", "Assumptions to check"] + _assumption_lines(diag, cfg, vendor, mode, direction_axis)

    blocking = any(diag.issues.get(cat) for cat in BLOCKING_ISSUES)
    if not has_output:
        status, code = "FAILED", EXIT_FAILED
        verdict = "FAILED: nothing was written. Fix the items above and run again."
    elif diag.dropped or diag.needs or blocking:
        status, code = "INCOMPLETE", EXIT_INCOMPLETE
        verdict = f"INCOMPLETE (exit code {code}): the output was written, but fix the items above for a correct trajectory."
    else:
        status, code = "COMPLETE", EXIT_COMPLETE
        verdict = "COMPLETE: every motion was used and nothing is missing. Check the assumptions above."
    lines.append("")
    lines += _wrap("Result     ", verdict)
    if diag.messages and warnings_hidden:
        lines.append(f"Details    {len(diag.messages)} warning(s); rerun with --verbose to list them")
    return ReadinessReport(status, code, "\n".join(lines), make_template(diag, vendor), verdict)


def make_template(diag: Diagnostics, vendor: str) -> Optional[str]:
    """YAML listing the missing controller data with TODO placeholders, or None if nothing config-able is missing."""
    rot = "abc" if vendor == "kuka" else "wpr"
    sections: dict[str, list[str]] = {}
    for n in diag.needs.values():
        if n.kind in ("tool", "base"):
            label = _tool_label(vendor, n.kind, n.key)
            hint = (f"from {label} in KRC:\\R1\\System\\$config.dat (X, Y, Z, A, B, C)" if vendor == "kuka" else
                    f"pendant: MENU > SETUP > Frames > {'Tool' if n.kind == 'tool' else 'User'} Frame (X, Y, Z, W, P, R)")
            sections.setdefault("tools" if n.kind == "tool" else "bases", []).extend([
                f"  # {label}, first used at {n.where or 'the program'}; {hint}",
                f"  {n.key}: {{xyz: [TODO, TODO, TODO], {rot}: [TODO, TODO, TODO]}}",
            ])
        elif n.kind == "position_register":
            sections.setdefault("fanuc.position_registers", []).extend([
                f"    # PR[{n.key}], first used at {n.where}; pendant: DATA > Position Reg. "
                "For a joint register use {joints: [J1, J2, J3, J4, J5, J6]}",
                f"    {n.key}: {{xyz: [TODO, TODO, TODO], wpr: [TODO, TODO, TODO]}}",
            ])
        elif n.kind == "register":
            sections.setdefault("fanuc.registers", []).extend([
                f"    # R[{n.key}], first used at {n.where}; pendant: DATA > Registers",
                f"    {n.key}: TODO",
            ])
        elif n.kind == "urdf":
            sections.setdefault("robot", []).extend([
                f"  # {n.count} joint-position move(s) need a model of your robot (first at {n.where}).",
                "  # Delete this section if you don't have one; those moves are then skipped.",
                "  urdf: TODO             # path to the robot's .urdf file",
                "  tip_link: TODO         # URDF link at the robot flange, e.g. tool0 (ABB/FANUC) or flange (KUKA)",
            ])
    if not sections:
        return None
    out = [
        "# robotparse: information your program uses that was not provided.",
        "# 1. Replace every TODO with the real value (millimetres and degrees),",
        "#    or delete the entries you don't have.",
        "# 2. Run again with this file added:  -c <this file>",
        "#    (it can be combined with your other config files; later files win).",
        "",
    ]
    if vendor == "kuka" and ("tools" in sections or "bases" in sections):
        out += ["# Alternative for tools and bases: load the controller's $config.dat instead:",
                "# kuka:", "#   include_dat: [path/to/$config.dat]", ""]
    for top in ("tools", "bases", "robot"):
        if top in sections:
            out += [f"{top}:"] + sections[top] + [""]
    fanuc = [k for k in ("fanuc.position_registers", "fanuc.registers") if k in sections]
    if fanuc:
        out.append("fanuc:")
        for k in fanuc:
            out += [f"  {k.split('.')[1]}:"] + sections[k]
        out.append("")
    return "\n".join(out)
