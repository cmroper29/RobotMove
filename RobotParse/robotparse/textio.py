"""Reading robot program files however they were saved (encoding, BOM, line endings)."""
from __future__ import annotations

import codecs
from pathlib import Path

from .model import RobotParseError

EXPORT_HINTS = {
    ".tp": "FANUC .tp programs are binary. Export the program as ASCII .ls instead (for example from "
           "ROBOGUIDE, or from the controller's MD: device, which lists every TP program as an .LS text file).",
    ".zip": "This is an archive. Extract it and pass the program files: KUKA .src + .dat (and "
            "KRC/R1/System/$config.dat for tools, bases and HOME), ABB .mod/.sys modules or the .pgf, "
            "FANUC .ls files.",
    ".pc": "This is a compiled FANUC KAREL program; only TP programs exported as .ls are supported.",
}


def read_program_text(path) -> str:
    path = Path(path)
    if not path.exists():
        raise RobotParseError(f"file not found: {path}")
    if path.is_dir():
        raise RobotParseError(f"{path} is a folder; pass the program files inside it")
    data = path.read_bytes()
    hint = EXPORT_HINTS.get(path.suffix.lower(), "")
    if data.startswith(b"PK\x03\x04"):
        raise RobotParseError(f"{path.name}: {EXPORT_HINTS['.zip']}")
    if data.startswith(codecs.BOM_UTF8):
        text = data[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace")
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16", errors="replace")
    else:
        utf16 = _utf16_without_bom(data)
        if utf16:
            text = data.decode(utf16, errors="replace")
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("cp1252", errors="replace")  # Windows editors / older controllers
    if _looks_binary(text):
        raise RobotParseError(f"{path.name} is not a text program file. {hint}".strip())
    return text.lstrip("﻿")


def _utf16_without_bom(data: bytes):
    sample = data[:4000]
    if len(sample) < 8:
        return None
    even, odd = sample[0::2], sample[1::2]
    zeros_even, zeros_odd = even.count(0) / len(even), odd.count(0) / len(odd)
    if zeros_odd > 0.3 and zeros_even < 0.05:
        return "utf-16-le"
    if zeros_even > 0.3 and zeros_odd < 0.05:
        return "utf-16-be"
    return None


def _looks_binary(text: str) -> bool:
    sample = text[:4000]
    if not sample:
        return False
    bad = sum(1 for ch in sample if (ord(ch) < 32 and ch not in "\t\n\r\f\x1a") or ch == "�")
    return bad / len(sample) > 0.05
