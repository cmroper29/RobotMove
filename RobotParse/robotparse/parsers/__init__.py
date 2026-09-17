"""Vendor detection and parser dispatch."""
from __future__ import annotations

import re
from pathlib import Path

from ..model import Diagnostics, Program, RobotParseError
from ..textio import EXPORT_HINTS, read_program_text

VENDOR_BY_SUFFIX = {
    ".src": "kuka", ".dat": "kuka", ".sub": "kuka",
    ".ls": "fanuc",
    ".mod": "abb", ".modx": "abb", ".prg": "abb", ".sys": "abb", ".sysx": "abb",
}
SUPPORTED = "KUKA .src/.dat, FANUC ASCII .ls, ABB .mod/.modx/.prg/.sys or a .pgf program file"


def expand_inputs(paths) -> list[Path]:
    """Check the input files exist and replace ABB .pgf program files by the modules they list."""
    out: list[Path] = []
    for p in map(Path, paths):
        if not p.exists():
            raise RobotParseError(f"file not found: {p}")
        if p.is_dir():
            raise RobotParseError(f"{p} is a folder; pass the program files inside it ({SUPPORTED})")
        if p.suffix.lower() == ".pgf":
            modules = re.findall(r"<Module>\s*([^<]+?)\s*</Module>", read_program_text(p))
            if not modules:
                raise RobotParseError(f"{p.name}: no <Module> entries found in the ABB program file")
            for name in modules:
                mod = p.parent / name
                if not mod.exists():
                    raise RobotParseError(f"{p.name} lists {name}, but that file is not next to it")
                out.append(mod)
        else:
            out.append(p)
    return out


def detect_vendor(path: Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in EXPORT_HINTS:
        raise RobotParseError(f"{path.name}: {EXPORT_HINTS[suffix]}")
    vendor = VENDOR_BY_SUFFIX.get(suffix)
    if vendor:
        return vendor
    if suffix == ".pgf":
        return "abb"
    head = read_program_text(path)[:4000].upper()
    if "/PROG" in head and "/MN" in head:
        return "fanuc"
    if ("MODULE" in head and "ENDMODULE" in head) or "%%%" in head:
        return "abb"
    if re.search(r"^\s*(GLOBAL\s+)?DEF(FCT|DAT)?\s", head, re.M):
        return "kuka"
    raise RobotParseError(f"cannot tell which robot language {path.name} is written in. "
                          f"Supported: {SUPPORTED}. Use --vendor if the file extension is unusual.")


def parse_program(paths, cfg: dict, vendor: str | None = None, diag: Diagnostics | None = None) -> Program:
    diag = diag or Diagnostics()
    vendor = (vendor or detect_vendor(Path(list(paths)[0]))).lower()
    paths = expand_inputs(paths)
    if vendor == "kuka":
        from .kuka import parse_krl
        program = parse_krl(paths, cfg, diag)
    elif vendor == "fanuc":
        from .fanuc import parse_ls
        program = parse_ls(paths, cfg, diag)
    elif vendor == "abb":
        from .abb import parse_rapid
        program = parse_rapid(paths, cfg, diag)
    else:
        raise RobotParseError(f"unsupported vendor {vendor!r} (expected kuka, fanuc or abb)")
    program.diag = diag
    return program
