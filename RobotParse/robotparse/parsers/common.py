"""Helpers shared by the vendor parsers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..model import Diagnostics

MAX_CALL_DEPTH = 32
MAX_LOOP_ITERATIONS = 10000


def strip_comment(line: str, marker: str, quotes: str = '"') -> str:
    """Remove a trailing comment that starts with `marker`, ignoring markers inside strings."""
    in_str: Optional[str] = None
    for i, ch in enumerate(line):
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in quotes:
            in_str = ch
        elif line.startswith(marker, i):
            return line[:i]
    return line


def split_top_level(text: str, sep: str = ",", quotes: str = '"') -> list[str]:
    """Split on `sep` outside brackets and strings. Empty items are kept."""
    parts, depth, start = [], 0, 0
    in_str: Optional[str] = None
    for i, ch in enumerate(text):
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in quotes:
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            start = i + len(sep)
    parts.append(text[start:])
    return [p.strip() for p in parts]


@dataclass
class Stmt:
    kind: str  # "stmt" or a control keyword: if/elseif/else/endif/for/endfor/while/endwhile/...
    text: str
    line: int
    file: str
    data: dict = field(default_factory=dict)


OPENERS = {
    "if": "endif",
    "for": "endfor",
    "while": "endwhile",
    "loop": "endloop",
    "repeat": "until",
    "switch": "endswitch",
}
BRANCHES = {"if": ("elseif", "else"), "switch": ("case", "default")}


class Routine:
    def __init__(self, name: str, stmts: list[Stmt], params: Optional[list[str]] = None, file: str = ""):
        self.name = name
        self.stmts = stmts
        self.params = params or []
        self.file = file
        self.ends: dict[int, int] = {}
        self.branches: dict[int, list[int]] = {}
        self._match()

    def _match(self) -> None:
        stack: list[int] = []
        for i, s in enumerate(self.stmts):
            if s.kind in OPENERS:
                stack.append(i)
                self.branches[i] = []
            elif s.kind in OPENERS.values():
                while stack and OPENERS[self.stmts[stack[-1]].kind] != s.kind:
                    stack.pop()  # tolerate malformed nesting
                if stack:
                    self.ends[stack.pop()] = i
            elif stack and s.kind in BRANCHES.get(self.stmts[stack[-1]].kind, ()):
                self.branches[stack[-1]].append(i)


class BlockRunner:
    """Walks routines statement by statement with a static view of control flow.

    Conditions are never evaluated (the program state is unknown offline): IF and
    SWITCH take their first branch, WHILE/LOOP/REPEAT bodies run once, and FOR loops
    with constant bounds are unrolled. Each simplification is reported once.
    """

    def __init__(self, diag: Diagnostics):
        self.diag = diag
        self.routines: dict[str, Routine] = {}
        self.depth = 0

    # vendor hooks -------------------------------------------------------
    def execute(self, stmt: Stmt) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def for_values(self, stmt: Stmt) -> Optional[list[float]]:
        return None

    def set_loop_var(self, stmt: Stmt, value: float) -> None:
        pass

    def bind_params(self, routine: Routine, args: list) -> None:
        pass

    # engine -------------------------------------------------------------
    def call(self, name: str, args: Optional[list] = None, where: str = "") -> bool:
        routine = self.routines.get(name.upper())
        if routine is None:
            return False
        if self.depth >= MAX_CALL_DEPTH:
            self.diag.warn(f"{where}: call depth limit reached calling {name}; call skipped")
            return True
        self.depth += 1
        try:
            self.bind_params(routine, args or [])
            self.run(routine, 0, len(routine.stmts))
        finally:
            self.depth -= 1
        return True

    def run(self, r: Routine, i0: int, i1: int) -> None:
        i = i0
        while i < i1:
            s = r.stmts[i]
            where = f"{s.file}:{s.line}"
            if s.kind in OPENERS:
                end = r.ends.get(i, i1)
                branches = r.branches.get(i, [])
                if s.kind == "if":
                    body_end = branches[0] if branches else end
                    if branches:
                        self.diag.warn(
                            f"{where}: IF conditions are not evaluated; using the first branch",
                            key=f"if:{where}",
                        )
                    self.run(r, i + 1, body_end)
                elif s.kind == "switch":
                    self.diag.warn(f"{where}: SWITCH/TEST is not evaluated; using the first case", key=f"sw:{where}")
                    if branches:
                        stop = branches[1] if len(branches) > 1 else end
                        self.run(r, branches[0] + 1, stop)
                elif s.kind == "for":
                    values = self.for_values(s)
                    if values is None:
                        self.diag.warn(f"{where}: FOR bounds are not constant; body executed once", key=f"for:{where}")
                        self.run(r, i + 1, end)
                    else:
                        if len(values) > MAX_LOOP_ITERATIONS:
                            self.diag.warn(f"{where}: FOR loop truncated to {MAX_LOOP_ITERATIONS} iterations")
                            values = values[:MAX_LOOP_ITERATIONS]
                        for v in values:
                            self.set_loop_var(s, v)
                            self.run(r, i + 1, end)
                else:
                    self.diag.warn(
                        f"{where}: {s.kind.upper()} loop condition is not evaluated; body executed once",
                        key=f"loop:{where}",
                    )
                    self.run(r, i + 1, end)
                i = end + 1
                continue
            if s.kind == "stmt":
                self.execute(s)
            i += 1


def frange(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        return []
    out, v = [], start
    while (step > 0 and v <= stop + 1e-9) or (step < 0 and v >= stop - 1e-9):
        out.append(v)
        v += step
        if len(out) > MAX_LOOP_ITERATIONS + 1:
            break
    return out


def safe(fn: Callable, default=None):
    try:
        return fn()
    except Exception:
        return default
