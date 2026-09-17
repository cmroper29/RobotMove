"""Helpers shared by the vendor parsers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..model import Diagnostics

MAX_CALL_DEPTH = 32
MAX_LOOP_ITERATIONS = 10000
MAX_WHILE_ITERATIONS = 1000  # WHILE/REPEAT loops whose condition stays true are cut off here


class ReturnSignal(Exception):
    """Raised by a RETURN statement to leave the current routine (with an optional value)."""

    def __init__(self, value=None):
        super().__init__()
        self.value = value


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
    def __init__(self, name: str, stmts: list[Stmt], params: Optional[list[str]] = None, file: str = "",
                 modes: Optional[list[str]] = None, is_function: bool = False):
        self.name = name
        self.stmts = stmts
        self.params = params or []
        self.modes = modes or ["OUT"] * len(self.params)  # IN = by value, OUT = by reference
        self.is_function = is_function
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
    """Walks routines statement by statement.

    Conditions are evaluated when the parser can compute them from known values
    (`eval_condition`). When a condition depends on run-time state it cannot know (inputs,
    variables without a value), IF takes that branch, WHILE/REPEAT bodies run once and LOOP
    bodies run once. SWITCH takes its first case, and FOR loops with constant bounds are
    unrolled. Each simplification is reported once.
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

    def eval_condition(self, stmt: Stmt) -> Optional[bool]:
        """True/False if the condition of an IF/ELSEIF/WHILE/UNTIL header can be computed, else None."""
        return None

    def eval_switch(self, switch: Stmt, cases: list[Stmt]) -> Optional[int]:
        """Index into `cases` (CASE/DEFAULT headers) of the branch to run, -1 for none, None if unknown."""
        return None

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
                    heads = [i] + branches
                    for head, stop in zip(heads, heads[1:] + [end]):
                        hs = r.stmts[head]
                        cond = True if hs.kind == "else" else self.eval_condition(hs)
                        if cond is None:
                            self._fallback(
                                f"{hs.file}:{hs.line}: condition cannot be evaluated offline "
                                f"('{hs.text[:60]}'); taking this branch",
                                key=f"if:{hs.file}:{hs.line}",
                            )
                        if cond is None or cond:
                            self.run(r, head + 1, stop)
                            break
                elif s.kind == "switch":
                    choice = self.eval_switch(s, [r.stmts[b] for b in branches])
                    if choice is None:
                        self._fallback(f"{where}: SWITCH/TEST value cannot be evaluated offline; using the first case",
                                       key=f"sw:{where}")
                        choice = 0
                    if branches and choice >= 0:
                        stop = branches[choice + 1] if choice + 1 < len(branches) else end
                        self.run(r, branches[choice] + 1, stop)
                elif s.kind == "for":
                    values = self.for_values(s)
                    if values is None:
                        self._fallback(f"{where}: FOR bounds are not constant; body executed once", key=f"for:{where}")
                        self.run(r, i + 1, end)
                    else:
                        if len(values) > MAX_LOOP_ITERATIONS:
                            self.diag.warn(f"{where}: FOR loop truncated to {MAX_LOOP_ITERATIONS} iterations")
                            values = values[:MAX_LOOP_ITERATIONS]
                        for v in values:
                            self.set_loop_var(s, v)
                            self.run(r, i + 1, end)
                elif s.kind in ("while", "repeat"):
                    self._conditional_loop(r, i, end)
                else:
                    self._fallback(f"{where}: endless LOOP; body executed once", key=f"loop:{where}")
                    self.run(r, i + 1, end)
                i = end + 1
                continue
            if s.kind == "stmt":
                self.execute(s)
            i += 1


    def _fallback(self, msg: str, key: str) -> None:
        """Warn about logic that depends on run-time state (counted once per program location)."""
        if self.diag.warn(msg, key=key):
            self.diag.counts["runtime_condition"] += 1

    def _conditional_loop(self, r: Routine, i: int, end: int) -> None:
        s = r.stmts[i]
        where = f"{s.file}:{s.line}"
        is_while = s.kind == "while"
        until = r.stmts[end] if not is_while and end < len(r.stmts) else None
        for n in range(MAX_WHILE_ITERATIONS + 1):
            if n == MAX_WHILE_ITERATIONS:
                self.diag.warn(f"{where}: loop stopped after {MAX_WHILE_ITERATIONS} iterations", key=f"loopcap:{where}")
                return
            if is_while:
                cond = self.eval_condition(s)
                if cond is None:
                    if n == 0:
                        self._fallback(f"{where}: WHILE condition cannot be evaluated offline; body executed once",
                                       key=f"loop:{where}")
                        self.run(r, i + 1, end)
                    else:
                        self.diag.warn(f"{where}: WHILE condition became unknown; loop stopped", key=f"loop:{where}")
                    return
                if not cond:
                    return
                self.run(r, i + 1, end)
            else:
                self.run(r, i + 1, end)
                cond = self.eval_condition(until) if until is not None else None
                if cond is None:
                    self._fallback(f"{where}: UNTIL condition cannot be evaluated offline; loop stopped",
                                   key=f"loop:{where}")
                    return
                if cond:
                    return


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
