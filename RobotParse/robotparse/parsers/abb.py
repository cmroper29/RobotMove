"""ABB RAPID parser (.mod, .modx, .prg, .sys)."""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lark import Lark, Token, Tree
from lark.exceptions import LarkError
from scipy.spatial.transform import Rotation

from ..model import (CartTarget, Diagnostics, Dwell, JointTarget, Motion, Program, RelativeTarget, RobotParseError,
                     Source, Speed, Zone)
from ..textio import read_program_text
from ..transforms import T_from_pos_quat, make_T, quat_wxyz_from_T, translation
from .common import BlockRunner, Routine, Stmt, frange, split_top_level, strip_comment

GRAMMAR = r"""
?expr: term
     | expr "+" term              -> add
     | expr "-" term              -> sub
?term: factor
     | term "*" factor            -> mul
     | term "/" factor            -> div
?factor: "-" factor               -> neg
       | "+" factor
       | postfix
?postfix: atom
        | postfix "." NAME        -> field
        | postfix "{" exprlist "}" -> index
?atom: NUMBER                     -> number
     | STRING                     -> string
     | NAME                       -> name
     | NAME "(" [args] ")"        -> call
     | "[" [exprlist] "]"         -> aggregate
     | "(" expr ")"
exprlist: expr ("," expr)*
args: item ("," item | optarg)*
?item: expr | optarg
optarg: "\\" NAME [":=" expr]
NAME: /[A-Za-z_][A-Za-z0-9_]*/
STRING: /"(?:[^"]|"")*"/
%import common.NUMBER
%import common.WS
%ignore WS
"""

_PARSER = Lark(GRAMMAR, parser="lalr", start=["expr", "args"], maybe_placeholders=True)

SCHEMA: dict[str, list[tuple[str, str]]] = {
    "robtarget": [("trans", "pos"), ("rot", "orient"), ("robconf", "confdata"), ("extax", "extjoint")],
    "jointtarget": [("robax", "robjoint"), ("extax", "extjoint")],
    "pos": [("x", "num"), ("y", "num"), ("z", "num")],
    "orient": [("q1", "num"), ("q2", "num"), ("q3", "num"), ("q4", "num")],
    "pose": [("trans", "pos"), ("rot", "orient")],
    "confdata": [("cf1", "num"), ("cf4", "num"), ("cf6", "num"), ("cfx", "num")],
    "robjoint": [(f"rax_{i}", "num") for i in range(1, 7)],
    "extjoint": [(f"eax_{c}", "num") for c in "abcdef"],
    "tooldata": [("robhold", "bool"), ("tframe", "pose"), ("tload", "loaddata")],
    "wobjdata": [("robhold", "bool"), ("ufprog", "bool"), ("ufmec", "string"), ("uframe", "pose"), ("oframe", "pose")],
    "speeddata": [("v_tcp", "num"), ("v_ori", "num"), ("v_leax", "num"), ("v_reax", "num")],
    "zonedata": [
        ("finep", "bool"), ("pzone_tcp", "num"), ("pzone_ori", "num"), ("pzone_eax", "num"),
        ("zone_ori", "num"), ("zone_leax", "num"), ("zone_reax", "num"),
    ],
    "loaddata": [("mass", "num"), ("cog", "pos"), ("aom", "orient"), ("ix", "num"), ("iy", "num"), ("iz", "num")],
}

_SPEEDS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500,
           3000, 4000, 5000, 6000, 7000]
_ZONES = {
    "z0": [0.3, 0.3, 0.3, 0.03, 0.3, 0.03], "z1": [1, 1, 1, 0.1, 1, 0.1], "z5": [5, 8, 8, 0.8, 8, 0.8],
    "z10": [10, 15, 15, 1.5, 15, 1.5], "z15": [15, 23, 23, 2.3, 23, 2.3], "z20": [20, 30, 30, 3, 30, 3],
    "z30": [30, 45, 45, 4.5, 45, 4.5], "z40": [40, 60, 60, 6, 60, 6], "z50": [50, 75, 75, 7.5, 75, 7.5],
    "z60": [60, 90, 90, 9, 90, 9], "z80": [80, 120, 120, 12, 120, 12], "z100": [100, 150, 150, 15, 150, 15],
    "z150": [150, 225, 225, 23, 225, 23], "z200": [200, 300, 300, 30, 300, 30],
}
_IDENTITY_POSE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]


class EvalError(Exception):
    pass


class UndefinedName(EvalError):
    def __init__(self, name: str):
        super().__init__(f"unknown data '{name}'")
        self.name = name


class ProgramExit(Exception):
    pass


@dataclass
class TV:
    """A typed RAPID value. `type` is the declared data type when known."""

    value: Any
    type: Optional[str] = None
    array: bool = False


@dataclass
class CurrentPos:
    """Placeholder for CRobT()/CJointT(): the robot position at run time."""

    delta: np.ndarray
    frame: str = "base"


def _predefined() -> dict[str, TV]:
    env = {f"v{v}": TV([float(v), 500.0, 5000.0, 1000.0], "speeddata") for v in _SPEEDS}
    env["vmax"] = TV([5000.0, 500.0, 5000.0, 1000.0], "speeddata")
    env["fine"] = TV([True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "zonedata")
    for name, vals in _ZONES.items():
        env[name] = TV([False, *map(float, vals)], "zonedata")
    load0 = [0.001, [0.0, 0.0, 0.001], [1.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0]
    env["load0"] = TV(load0, "loaddata")
    env["tool0"] = TV([True, copy.deepcopy(_IDENTITY_POSE), load0], "tooldata")
    env["wobj0"] = TV([False, True, "", copy.deepcopy(_IDENTITY_POSE), copy.deepcopy(_IDENTITY_POSE)], "wobjdata")
    env["true"] = TV(True, "bool")
    env["false"] = TV(False, "bool")
    env["pi"] = TV(math.pi, "num")
    return env


def _is_vec(v, n) -> bool:
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def value_kind(tv: TV) -> Optional[str]:
    if tv is None:
        return None
    if tv.type in SCHEMA and not tv.array:
        return tv.type
    v = tv.value
    if isinstance(v, CurrentPos):
        return "robtarget"
    if isinstance(v, list):
        if len(v) == 4 and _is_vec(v[0], 3) and _is_vec(v[1], 4):
            return "robtarget"
        if len(v) == 2 and isinstance(v[0], list) and len(v[0]) == 6:
            return "jointtarget"
        if _is_vec(v, 4):
            return "speeddata"
        if len(v) == 7 and isinstance(v[0], bool):
            return "zonedata"
        if len(v) == 3 and isinstance(v[0], bool) and isinstance(v[1], list):
            return "tooldata"
        if len(v) == 5 and isinstance(v[0], bool):
            return "wobjdata"
    return None


def pose_to_T(pose) -> np.ndarray:
    return T_from_pos_quat(pose[0], pose[1])


def T_to_pose(T) -> list:
    return [list(map(float, T[:3, 3])), list(map(float, quat_wxyz_from_T(T)))]


HEADERS = [
    ("module", r"(?:LOCAL\s+|NOVIEW\s+)*MODULE\s+(\w+).*"),
    ("endmodule", r"ENDMODULE"),
    ("proc", r"(?:LOCAL\s+)?PROC\s+(\w+)\s*\((.*)\)"),
    ("proc", r"(?:LOCAL\s+)?FUNC\s+\w+\s+(\w+)\s*\((.*)\)"),
    ("proc", r"(?:LOCAL\s+)?TRAP\s+(\w+)()"),
    ("endroutine", r"END(?:PROC|FUNC|TRAP)"),
    ("if", r"IF\s+(.+)\s+THEN"),
    ("elseif", r"ELSEIF\s+(.+)\s+THEN"),
    ("else", r"ELSE"),
    ("endif", r"ENDIF"),
    ("for", r"FOR\s+(\w+)\s+FROM\s+(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+?))?\s+DO"),
    ("endfor", r"ENDFOR"),
    ("while", r"WHILE\s+(.+)\s+DO"),
    ("endwhile", r"ENDWHILE"),
    ("switch", r"TEST\s+(.+)"),
    ("case", r"CASE\s+(.+):"),
    ("default", r"DEFAULT\s*:"),
    ("endswitch", r"ENDTEST"),
    ("handler", r"(?:ERROR|UNDO|BACKWARD)(?:\s*\(.*\))?"),
]
_HEADER_RES = [(k, re.compile(p, re.I | re.S)) for k, p in HEADERS]

DECL_RE = re.compile(
    r"^(?:LOCAL\s+|TASK\s+)*(CONST|PERS|VAR)\s+(\w+)\s+(\w+)\s*(\{[^}]*\})?\s*(?::=\s*(.*))?$", re.I | re.S
)
ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*(?:\s*\{[^}]*\})?(?:\s*\.\s*\w+(?:\s*\{[^}]*\})?)*)\s*:=\s*(.*)$", re.S)
INSTR_RE = re.compile(r"^([A-Za-z_]\w*)\s*(.*)$", re.S)
MOTION_RE = re.compile(
    r"^(?:(?P<absj>moveabsj)|(?:move|trigg|search|arc|cap|spot|paint|disp)(?P<ip>[jlc]))(?!ext)\w*$", re.I
)
IGNORED = {
    "confj", "confl", "singarea", "setdo", "setgo", "setao", "reset", "set", "pulsedo", "tpwrite", "tperase",
    "stop", "break", "waitdi", "waitdo", "waituntil", "gripload", "motionsup", "pathaccLim".lower(), "worldacclim",
    "stoppmove", "startmove", "storepath", "restopath", "clearpath", "incr", "decr", "clkstart", "clkstop",
    "clkreset", "errwrite", "connect", "idelete", "isignaldi", "itimer", "open", "close", "write", "movextj",
}


def segment(text: str, file: str) -> list[Stmt]:
    """Split RAPID source into statements and control-structure headers."""
    stmts: list[Stmt] = []
    pending, start = "", 0
    skipping_header = False
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if s == "%%%":
            skipping_header = not skipping_header
            continue
        if skipping_header:
            continue
        s = strip_comment(s, "!").strip()
        if not s:
            continue
        if not pending and not s.endswith(";"):
            for kind, rx in _HEADER_RES:
                m = rx.fullmatch(s)
                if m:
                    stmts.append(Stmt(kind, s, ln, file, {"groups": m.groups()}))
                    break
            else:
                pending, start = s, ln
            continue
        if not pending:
            start = ln
        pending = f"{pending} {s}".strip()
        while True:
            parts = split_top_level(pending, ";")
            if len(parts) == 1:
                break
            stmt_text = parts[0]
            if stmt_text:
                stmts.append(Stmt("stmt", stmt_text, start, file))
            pending = ";".join(parts[1:]).strip()
            start = ln
    return stmts


class RapidParser(BlockRunner):
    def __init__(self, cfg: dict, diag: Diagnostics):
        super().__init__(diag)
        self.cfg = cfg
        self.globals: dict[str, TV] = _predefined()
        self.frames: list[dict[str, TV]] = []
        self.commands: list = []
        self.acc_pct = 100.0
        self.vel_override = 100.0
        self.vel_max = float("inf")
        self.module_decls: list[Stmt] = []
        self.routine_order: list[str] = []
        self.unresolved: list[str] = []  # undefined data names met while evaluating the current statement

    # ------------------------------------------------------------------ loading
    def load(self, path: Path) -> None:
        stmts = segment(read_program_text(path), path.name)
        current: Optional[tuple[str, list[str], list[Stmt]]] = None
        in_handler = False
        for s in stmts:
            if s.kind == "proc":
                name, params = s.data["groups"][0], s.data["groups"][1]
                current = (name, self._param_names(params), [])
                in_handler = False
            elif s.kind == "endroutine":
                if current:
                    self.routines[current[0].upper()] = Routine(current[0], current[2], current[1], path.name)
                    self.routine_order.append(current[0].upper())
                current = None
            elif s.kind == "handler":
                in_handler = True
            elif s.kind in ("module", "endmodule"):
                continue
            elif current is not None:
                if not in_handler:
                    current[2].append(s)
            elif s.kind == "stmt":
                self.module_decls.append(s)

    @staticmethod
    def _param_names(text: str) -> list[str]:
        names = []
        for p in split_top_level(text, ","):
            p = p.strip()
            if not p or p.startswith("\\"):
                continue
            names.append(p.split()[-1].split("{")[0].lower())
        return names

    def declare_module_data(self) -> None:
        pending = list(self.module_decls)
        for _ in range(3):  # retry to tolerate forward references
            failed = []
            for s in pending:
                try:
                    self._declare(s, self.globals)
                except (EvalError, LarkError):
                    failed.append(s)
            if not failed or len(failed) == len(pending):
                pending = failed
                break
            pending = failed
        for s in pending:
            self.diag.warn(f"{s.file}:{s.line}: could not evaluate declaration: {s.text[:80]}")

    # ------------------------------------------------------------------ environment
    def lookup(self, name: str) -> TV:
        key = name.lower()
        for frame in reversed(self.frames):
            if key in frame:
                return frame[key]
        if key in self.globals:
            return self.globals[key]
        raise UndefinedName(name)

    def _scope_for(self, key: str) -> dict:
        for frame in reversed(self.frames):
            if key in frame:
                return frame
        return self.globals

    def _declare(self, s: Stmt, scope: dict) -> bool:
        m = DECL_RE.match(s.text)
        if not m:
            return False
        _, typ, name, dims, value = m.groups()
        typ = typ.lower()
        if value is not None:
            tv = self.eval_text(value)
            scope[name.lower()] = TV(tv.value, typ, array=bool(dims))
        else:
            scope[name.lower()] = TV(None, typ, array=bool(dims))
        return True

    # ------------------------------------------------------------------ evaluation
    def eval_text(self, text: str) -> TV:
        return self.eval(_PARSER.parse(text, start="expr"))

    def eval_args(self, text: str) -> tuple[list[Optional[TV]], dict[str, Optional[TV]]]:
        if not text.strip():
            return [], {}
        return self._args(_PARSER.parse(text, start="args"), tolerant=True)

    def _args(self, node: Optional[Tree], tolerant: bool = False):
        pos: list[Optional[TV]] = []
        opt: dict[str, Optional[TV]] = {}
        if node is None:
            return pos, opt
        items = node.children if isinstance(node, Tree) and node.data == "args" else [node]
        for item in items:
            if isinstance(item, Tree) and item.data == "optarg":
                name, expr = item.children
                val = None
                if expr is not None:
                    try:
                        val = self.eval(expr)
                    except EvalError as exc:
                        if not tolerant:
                            raise
                        if isinstance(exc, UndefinedName):
                            self.unresolved.append(exc.name)
                opt[str(name).lower()] = val if expr is not None else TV(True, "switch")
            else:
                try:
                    pos.append(self.eval(item))
                except EvalError as exc:
                    if not tolerant:
                        raise
                    if isinstance(exc, UndefinedName):
                        self.unresolved.append(exc.name)
                    pos.append(None)
        return pos, opt

    def eval(self, node) -> TV:
        if isinstance(node, Token):
            raise EvalError(f"unexpected token {node}")
        d = node.data
        ch = node.children
        if d == "number":
            return TV(float(ch[0]), "num")
        if d == "string":
            return TV(str(ch[0])[1:-1].replace('""', '"'), "string")
        if d == "name":
            return self.lookup(str(ch[0]))
        if d == "aggregate":
            items = ch[0].children if ch[0] is not None else []
            return TV([self.eval(c).value for c in items])
        if d in ("add", "sub", "mul", "div"):
            a, b = self.eval(ch[0]).value, self.eval(ch[1]).value
            return TV(_arith(d, a, b))
        if d == "neg":
            return TV(_arith("mul", -1.0, self.eval(ch[0]).value))
        if d == "field":
            base = self.eval(ch[0])
            return _field(base, str(ch[1]).lower())
        if d == "index":
            base = self.eval(ch[0])
            idx = [int(round(self.eval(c).value)) for c in ch[1].children]
            val = base.value
            for i in idx:
                if not isinstance(val, list) or not 1 <= i <= len(val):
                    raise EvalError("array index out of range")
                val = val[i - 1]
            return TV(val, base.type, array=False)
        if d == "call":
            return self._call(str(ch[0]), ch[1])
        raise EvalError(f"unsupported expression '{d}'")

    def _call(self, name: str, args_node) -> TV:
        n = name.lower()
        pos, opt = self._args(args_node)
        vals = [p.value if p is not None else None for p in pos]
        if n in ("crobt", "cpos"):
            return TV(CurrentPos(np.eye(4)), "robtarget")
        if n == "cjointt":
            return TV(CurrentPos(np.eye(4), frame="joints"), "jointtarget")
        if n == "offs":
            p, dx, dy, dz = vals[:4]
            if isinstance(p, CurrentPos):
                return TV(CurrentPos(translation(dx, dy, dz) @ p.delta, "base"), "robtarget")
            out = copy.deepcopy(p)
            out[0] = [out[0][0] + dx, out[0][1] + dy, out[0][2] + dz]
            return TV(out, "robtarget")
        if n == "reltool":
            p, dx, dy, dz = vals[:4]
            angles = [opt[k].value if opt.get(k) is not None else 0.0 for k in ("rx", "ry", "rz")]
            delta = translation(dx, dy, dz) @ make_T([0, 0, 0], Rotation.from_euler("XYZ", angles, degrees=True))
            if isinstance(p, CurrentPos):
                return TV(CurrentPos(p.delta @ delta, "tool"), "robtarget")
            out = copy.deepcopy(p)
            T = pose_to_T(out[:2]) @ delta
            out[0], out[1] = T_to_pose(T)
            return TV(out, "robtarget")
        if n == "orientzyx":
            z, y, x = vals[:3]
            q = quat_wxyz_from_T(make_T([0, 0, 0], Rotation.from_euler("ZYX", [z, y, x], degrees=True)))
            return TV(list(map(float, q)), "orient")
        if n == "posemult":
            return TV(T_to_pose(pose_to_T(vals[0]) @ pose_to_T(vals[1])), "pose")
        if n == "poseinv":
            return TV(T_to_pose(np.linalg.inv(pose_to_T(vals[0]))), "pose")
        if n == "norient":
            q = np.asarray(vals[0], float)
            return TV(list(q / np.linalg.norm(q)), "orient")
        math_fns = {
            "sqrt": math.sqrt, "abs": abs, "sin": lambda a: math.sin(math.radians(a)),
            "cos": lambda a: math.cos(math.radians(a)), "tan": lambda a: math.tan(math.radians(a)),
            "atan": lambda a: math.degrees(math.atan(a)), "asin": lambda a: math.degrees(math.asin(a)),
            "acos": lambda a: math.degrees(math.acos(a)), "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
            "exp": math.exp, "pow": math.pow, "trunc": lambda a, *_: float(math.trunc(a)),
            "round": lambda a, *_: float(round(a)),
        }
        if n in math_fns:
            return TV(float(math_fns[n](*vals)), "num")
        raise EvalError(f"unsupported function {name}()")

    # ------------------------------------------------------------------ execution
    def for_values(self, s: Stmt):
        var, a, b, step = s.data["groups"]
        try:
            return frange(self.eval_text(a).value, self.eval_text(b).value, self.eval_text(step).value if step else 1.0)
        except (EvalError, LarkError, TypeError):
            return None

    def set_loop_var(self, s: Stmt, value: float) -> None:
        scope = self.frames[-1] if self.frames else self.globals
        scope[s.data["groups"][0].lower()] = TV(float(value), "num")

    def bind_params(self, routine: Routine, args: list) -> None:
        frame = self.frames[-1]
        for name, val in zip(routine.params, args):
            if val is not None:
                frame[name] = val

    def call(self, name: str, args=None, where: str = "") -> bool:
        if name.upper() not in self.routines:
            return False
        self.frames.append({})
        try:
            return super().call(name, args, where)
        finally:
            self.frames.pop()

    def run_program(self, entry: Optional[str]) -> None:
        candidates = [entry] if entry else ["main"]
        for name in candidates:
            if name and name.upper() in self.routines:
                break
        else:
            if entry:
                raise RobotParseError(f"entry routine '{entry}' not found; routines in the loaded modules: "
                                      + ", ".join(self.routines[r].name for r in self.routine_order))
            if not self.routine_order:
                raise RobotParseError("no PROC found: is this an ABB RAPID module (MODULE ... ENDMODULE)?")
            name = self.routines[self.routine_order[0]].name
            self.diag.warn(f"no main routine; starting at {name}")
        try:
            self.call(name)
        except ProgramExit:
            pass

    def execute(self, s: Stmt) -> None:
        where = f"{s.file}:{s.line}"
        self.unresolved = []
        is_motion = False
        try:
            if DECL_RE.match(s.text):
                self._declare(s, self.frames[-1] if self.frames else self.globals)
                return
            m = ASSIGN_RE.match(s.text)
            if m:
                self._assign(m.group(1), self.eval_text(m.group(2)))
                return
            m = INSTR_RE.match(s.text)
            if not m:
                return
            name, rest = m.group(1), m.group(2)
            low = name.lower()
            if MOTION_RE.match(name) and not low.startswith("moveext"):
                is_motion = True
                self.diag.motion_statements += 1
                self._motion(name, rest, s)
            elif low == "waittime":
                pos, _ = self.eval_args(rest)
                if pos and pos[-1] is not None:
                    self.commands.append(Dwell(float(pos[-1].value), Source(s.file, s.line, s.text)))
            elif low == "accset":
                pos, _ = self.eval_args(rest)
                if pos and pos[0] is not None:
                    self.acc_pct = float(pos[0].value)
            elif low == "velset":
                pos, _ = self.eval_args(rest)
                if len(pos) >= 2 and pos[0] is not None and pos[1] is not None:
                    self.vel_override, self.vel_max = float(pos[0].value), float(pos[1].value)
            elif low == "exit":
                raise ProgramExit()
            elif low == "return":
                self.diag.warn(f"{where}: RETURN ignored (static execution)", key=f"ret:{where}")
            elif name.upper() in self.routines:
                pos, _ = self.eval_args(rest)
                self.call(name, pos, where)
            elif low in IGNORED:
                return
            else:
                self.diag.issue("ignored_instruction", name)
                self.diag.warn(f"{where}: instruction '{name}' ignored", key=f"instr:{low}")
        except ProgramExit:
            raise
        except (EvalError, LarkError, TypeError, IndexError, ValueError) as exc:
            if isinstance(exc, UndefinedName):
                self.unresolved.append(exc.name)
            for n in self.unresolved:
                self.diag.need("name", n, where)
            if is_motion:
                self.diag.drop(where, str(exc).split("\n")[0])
            else:
                self.diag.warn(f"{where}: could not interpret '{s.text[:80]}': {exc}".split("\n")[0])

    def _assign(self, lhs: str, value: TV) -> None:
        tree = _PARSER.parse(lhs, start="expr")
        chain = []
        while tree.data in ("field", "index"):
            chain.append(tree)
            tree = tree.children[0]
        if tree.data != "name":
            raise EvalError("unsupported assignment target")
        key = str(tree.children[0]).lower()
        scope = self._scope_for(key)
        if not chain:
            old = scope.get(key)
            scope[key] = TV(copy.deepcopy(value.value), old.type if old else value.type, old.array if old else False)
            return
        root = scope.get(key)
        if root is None:
            raise EvalError(f"unknown data '{key}'")
        container, typ = root.value, root.type
        chain.reverse()
        for i, node in enumerate(chain):
            last = i == len(chain) - 1
            if node.data == "field":
                fname = str(node.children[1]).lower()
                idx, typ = _field_index(typ, fname, container)
            else:
                idx = int(round(self.eval(node.children[1].children[0]).value)) - 1
            if last:
                container[idx] = copy.deepcopy(value.value)
            else:
                container = container[idx]

    def _motion(self, name: str, rest: str, s: Stmt) -> None:
        src = Source(s.file, s.line, s.text)
        where = str(src)
        m = MOTION_RE.match(name)
        kind = "PTP" if m.group("absj") else {"j": "PTP", "l": "LIN", "c": "CIRC"}[m.group("ip").lower()]
        pos, opt = self.eval_args(rest)
        buckets: dict[str, list[TV]] = {}
        for tv in pos:
            k = value_kind(tv)
            if k:
                buckets.setdefault(k, []).append(tv)
        missing = [k for k in ("speeddata", "zonedata", "tooldata") if k not in buckets]
        if name.lower().startswith("search"):
            missing = [k for k in missing if k != "zonedata"]
        targets = buckets.get("jointtarget" if m.group("absj") else "robtarget", [])
        wobj_unresolved = "wobj" in opt and opt["wobj"] is None
        if missing or not targets or wobj_unresolved:
            for n in self.unresolved:
                self.diag.need("name", n, where)
        if missing:
            self.diag.warn(f"{where}: could not resolve {', '.join(missing)} for {name} (undefined data?); "
                           "using v1000/fine/tool0 for them")
        if wobj_unresolved:
            self.diag.warn(f"{where}: work object of {name} is undefined; using wobj0")
        if not targets:
            self.diag.drop(where, "uses data that is not defined in the loaded files" if self.unresolved
                           else f"the target of {name} could not be resolved")
            return
        target = self._target(targets[-1])
        via = self._target(targets[0]) if kind == "CIRC" and len(targets) >= 2 else None
        if kind == "CIRC" and via is None:
            self.diag.warn(f"{where}: {name} without circle point; treated as linear")
            kind = "LIN"

        sp = buckets.get("speeddata", [TV(self.globals["v1000"].value)])[0].value
        speed = Speed(tcp=float(sp[0]), ori=float(sp[1]), accel_pct=self.acc_pct)
        if opt.get("v") is not None:
            speed.tcp = float(opt["v"].value)
        if opt.get("t") is not None:
            speed.duration = float(opt["t"].value)
        speed.tcp = min(speed.tcp * self.vel_override / 100.0, self.vel_max)

        zones = buckets.get("zonedata")
        if zones:
            zd = zones[0].value
            zone = Zone() if zd[0] else Zone.blended(dist=float(zd[1]), ori=float(zd[4]))
        else:
            zone = Zone()
        if opt.get("z") is not None and not zone.fine:
            zone.dist = float(opt["z"].value)

        tools = buckets.get("tooldata")
        tool = np.eye(4)
        if tools:
            td = tools[0].value
            if not td[0]:
                self.diag.warn(f"{where}: stationary tool (robhold FALSE) is not supported; using it as a robot-held tool")
            tool = pose_to_T(td[1])
        base = np.eye(4)
        wobj = opt.get("wobj")
        if wobj is not None:
            wd = wobj.value
            if wd[0]:
                self.diag.warn(f"{where}: robot-held work object (robhold TRUE) is not supported")
            base = pose_to_T(wd[3]) @ pose_to_T(wd[4])
        self.commands.append(Motion(kind, target, speed, zone, tool, base, src, via=via))

    def _target(self, tv: TV):
        v = tv.value
        if isinstance(v, CurrentPos):
            return RelativeTarget(v.delta, v.frame if v.frame in ("base", "tool") else "base")
        if value_kind(tv) == "jointtarget":
            return JointTarget(np.asarray(v[0], dtype=float))
        T = pose_to_T(v[:2])
        cf = v[2] if len(v) > 2 else None
        return CartTarget(T, {"abb_cf": [int(c) for c in cf]} if _is_vec(cf, 4) else {})


def _arith(op: str, a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_arith(op, x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_arith(op, x, b) for x in a]
    if isinstance(b, list):
        return [_arith(op, a, y) for y in b]
    return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]


def _field_index(typ: Optional[str], fname: str, value) -> tuple[int, Optional[str]]:
    if typ in SCHEMA:
        for i, (f, sub) in enumerate(SCHEMA[typ]):
            if f == fname:
                return i, sub
    for t, fields in SCHEMA.items():  # type unknown: guess by field name and shape
        for i, (f, sub) in enumerate(fields):
            if f == fname and (not isinstance(value, list) or len(value) == len(fields)):
                return i, sub
    raise EvalError(f"unknown field '{fname}'")


def _field(base: TV, fname: str) -> TV:
    idx, sub = _field_index(base.type, fname, base.value)
    if not isinstance(base.value, list) or idx >= len(base.value):
        raise EvalError(f"cannot access field '{fname}'")
    return TV(base.value[idx], sub)


def parse_rapid(paths: list[Path], cfg: dict, diag: Diagnostics) -> Program:
    parser = RapidParser(cfg, diag)
    for p in paths:
        parser.load(Path(p))
    parser.declare_module_data()
    parser.run_program(cfg.get("entry"))
    return Program("abb", Path(paths[0]).stem, parser.commands, list(diag.messages))
