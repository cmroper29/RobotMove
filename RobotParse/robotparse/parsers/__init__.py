"""Vendor detection and parser dispatch."""
from __future__ import annotations

from pathlib import Path

from ..model import Diagnostics, Program

VENDOR_BY_SUFFIX = {
    ".src": "kuka", ".dat": "kuka", ".sub": "kuka",
    ".ls": "fanuc",
    ".mod": "abb", ".modx": "abb", ".prg": "abb", ".sys": "abb", ".sysx": "abb",
}


def detect_vendor(path: Path) -> str:
    vendor = VENDOR_BY_SUFFIX.get(path.suffix.lower())
    if vendor:
        return vendor
    head = path.read_text(errors="replace")[:4000].upper()
    if "/PROG" in head and "/MN" in head:
        return "fanuc"
    if "MODULE" in head and "ENDMODULE" in head or "%%%" in head:
        return "abb"
    if "DEF " in head or "DEFDAT" in head:
        return "kuka"
    raise ValueError(f"cannot detect robot language of {path}; pass --vendor")


def parse_program(paths, cfg: dict, vendor: str | None = None, diag: Diagnostics | None = None) -> Program:
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    diag = diag or Diagnostics()
    vendor = (vendor or detect_vendor(paths[0])).lower()
    if vendor == "kuka":
        from .kuka import parse_krl
        return parse_krl(paths, cfg, diag)
    if vendor == "fanuc":
        from .fanuc import parse_ls
        return parse_ls(paths, cfg, diag)
    if vendor == "abb":
        from .abb import parse_rapid
        return parse_rapid(paths, cfg, diag)
    raise ValueError(f"unsupported vendor {vendor!r} (expected kuka, fanuc or abb)")
