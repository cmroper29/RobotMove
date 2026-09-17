"""KUKA KRL parser (.src with its .dat, plus optional $config.dat)."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lark import Lark, Token, Tree
from lark.exceptions import LarkError

from ..model import (CartTarget, Diagnostics, Dwell, JointTarget, Motion, Program, RelativeTarget, RobotParseError,
                     Source, Speed, Zone)
from ..textio import read_program_text
from ..transforms import T_from_xyzabc, frame_from_spec, xyzabc_from_T
from .common import (MAX_CALL_DEPTH, BlockRunner, ReturnSignal, Routine, Stmt, frange, split_top_level,
                     strip_comment)

GRAMMAR = r"""
?expr: cmp
?cmp: or_expr
    | cmp "==" or_expr           -> eq
    | cmp "<>" or_expr           -> ne
    | cmp "<=" or_expr           -> le
    | cmp ">=" or_expr           -> ge
    | cmp "<" or_expr            -> lt
    | cmp ">" or_expr            -> gt
?or_expr: exor_expr
    | or_expr "OR" exor_expr     -> or_
?exor_expr: and_expr
    | exor_expr "EXOR" and_expr  -> xor
?and_expr: sum
    | and_expr "AND" sum         -> and_
?sum: product
    | sum "+" product            -> add
    | sum "-" product            -> sub
?product: geo
    | product "*" geo            -> mul
    | product "/" geo            -> div
?geo: unary
    | geo ":" unary              -> geo
?unary: "-" unary                -> neg
      | "+" unary
      | "NOT" unary              -> not_
      | postfix
?postfix: atom
        | postfix "." NAME       -> field
        | postfix "[" [exprlist] "]" -> index
?atom: NUMBER                    -> number
     | STRING                    -> string
     | "#" NAME                  -> enum
     | NAME                      -> name
     | NAME "(" callargs ")"     -> call
     | "{" NAME ":" [fields] "}" -> typed_struct
     | "{" [fields] "}"          -> struct
     | "(" expr ")"
exprlist: expr ("," expr)*
callargs: [expr] ("," [expr])*
fields: field ("," field)*
field: NAME expr
     | NAME "[" "]" expr
NAME: /\$?[A-Z_][A-Z0-9_$]*/
STRING: /"[^"]*"/
%import common.NUMBER
%import common.WS
%ignore WS
"""

_PARSER = Lark(GRAMMAR, parser="lalr", start="expr", maybe_placeholders=True)

CART_KEYS = ("X", "Y", "Z", "A", "B", "C")
AXIS_KEYS = tuple(f"A{i}" for i in range(1, 7))
BUILTIN_TYPES = {"E6POS", "POS", "FRAME", "E6AXIS", "AXIS", "FDAT", "PDAT", "LDAT", "INT", "REAL", "BOOL", "CHAR", "LOAD"}

HEADERS = [
    ("def", r"(?:GLOBAL\s+)?DEF\s+(\$?\w+)\s*\((.*)\)"),
    ("deffct", r"(?:GLOBAL\s+)?DEFFCT\s+(\w+(?:\s*\[\s*\d*\s*\])?)\s+(\$?\w+)\s*\((.*)\)"),
    ("end", r"END"),
    ("endfct", r"ENDFCT"),
    ("defdat", r"DEFDAT\s+.*"),
    ("enddat", r"ENDDAT"),
    ("if", r"IF\s+(.+)\s+THEN"),
    ("else", r"ELSE"),
    ("endif", r"ENDIF"),
    ("for", r"FOR\s+(\$?\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?"),
    ("endfor", r"ENDFOR"),
    ("while", r"WHILE\s+(.+)"),
    ("endwhile", r"ENDWHILE"),
    ("loop", r"LOOP"),
    ("endloop", r"ENDLOOP"),
    ("repeat", r"REPEAT"),
    ("until", r"UNTIL\s+(.+)"),
    ("switch", r"SWITCH\s+(.+)"),
    ("case", r"CASE\s+(.+)"),
    ("default", r"DEFAULT"),
    ("endswitch", r"ENDSWITCH"),
]
_HEADER_RES = [(k, re.compile(p)) for k, p in HEADERS]

DECL_RE = re.compile(r"^DECL\s+(?:GLOBAL\s+)?(?:CONST\s+)?(\w+)\s+(.+)$")
TYPED_DECL_RE = re.compile(r"^(?:GLOBAL\s+)?(?:CONST\s+)?(" + "|".join(sorted(BUILTIN_TYPES)) + r")\s+(\$?\w+.*)$")
ASSIGN_RE = re.compile(r"^(\$?\w+(?:\[[^\]]*\])?(?:\.\w+(?:\[[^\]]*\])?)*)\s*=\s*(.+)$")
MOTION_RE = re.compile(r"^(PTP_REL|LIN_REL|CIRC_REL|SPTP_REL|SLIN_REL|SCIRC_REL|PTP|LIN|CIRC|SPTP|SLIN|SCIRC|SPL)\b\s*(.*)$")
CALL_RE = re.compile(r"^(\$?\w+)\s*\((.*)\)$")
APPROX_RE = re.compile(r"\s+(C_PTP|C_DIS|C_VEL|C_ORI|C_SPL|#BASE|#TOOL)\s*$")
WITH_RE = re.compile(r"\s+WITH\s+")
RETURN_RE = re.compile(r"^RETURN(?:\s+(.+))?$")


class EvalError(Exception):
    pass


class UndefinedName(EvalError):
    def __init__(self, name: str):
        super().__init__(f"unknown variable {name}")
        self.name = name


class _DefaultHome(dict):
    """XHOME when no $config.dat was loaded (factory default)."""


class KArray(dict):
    """Sparse KRL array keyed by index tuples."""


class _Unknown:
    """Value of a variable that is declared but only gets its value at run time."""

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()
_NO_RETURN = object()
EVAL_ERRORS = (EvalError, LarkError, TypeError, ValueError, KeyError, ZeroDivisionError)


def _is_lvalue(node) -> bool:
    while isinstance(node, Tree) and node.data in ("field", "index"):
        node = node.children[0]
    return isinstance(node, Tree) and node.data == "name"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def frame_to_T(v) -> np.ndarray:
    if not isinstance(v, dict):
        raise EvalError("expected a frame/position structure")
    return T_from_xyzabc(*(float(v.get(k, 0.0)) for k in CART_KEYS))


def T_to_frame(T) -> dict:
    return dict(zip(CART_KEYS, map(float, xyzabc_from_T(T))))


class KrlParser(BlockRunner):
    def __init__(self, cfg: dict, diag: Diagnostics):
        super().__init__(diag)
        self.cfg = cfg
        self.env: dict[str, Any] = {}
        self.locals: list[dict[str, Any]] = []
        self.commands: list = []
        self.routine_order: list[str] = []
        self.last_cart: Optional[dict] = None
        self.last_axes: Optional[dict] = None
        self.spline_block: Optional[int] = None  # index into commands where a SPLINE block started
        self.where = ""
        self._init_system()

    # ------------------------------------------------------------------ state
    def _init_system(self) -> None:
        m = self.cfg["motion"]
        null = dict.fromkeys(CART_KEYS, 0.0)
        self.env.update({
            "$NULLFRAME": null,
            "$TOOL": dict(null),
            "$BASE": dict(null),
            "$VEL": {"CP": m["default_tcp_speed"] / 1000.0, "ORI1": m["default_ori_speed"], "ORI2": m["default_ori_speed"]},
            "$ACC": {"CP": m["cart_accel"] / 1000.0, "ORI1": 1000.0, "ORI2": 1000.0},
            "$APO": {"CDIS": 0.0, "CPTP": 0.0, "CVEL": 0.0, "CORI": 0.0},
            "$VEL_AXIS": KArray({(i,): 100.0 for i in range(1, 7)}),
            "$ACC_AXIS": KArray({(i,): 100.0 for i in range(1, 7)}),
            "$OV_PRO": 100.0,
            "TOOL_DATA": KArray(),
            "BASE_DATA": KArray(),
            "XHOME": _DefaultHome({"A1": 0.0, "A2": -90.0, "A3": 90.0, "A4": 0.0, "A5": 0.0, "A6": 0.0}),
            "TRUE": True,
            "FALSE": False,
        })

    def apply_config_frames(self) -> None:
        for n, spec in self.cfg.get("tools", {}).items():
            self.env["TOOL_DATA"][(int(n),)] = T_to_frame(frame_from_spec(spec))
        for n, spec in self.cfg.get("bases", {}).items():
            self.env["BASE_DATA"][(int(n),)] = T_to_frame(frame_from_spec(spec))

    def lookup(self, name: str):
        for frame in reversed(self.locals):
            if name in frame:
                value = frame[name]
                break
        else:
            if name not in self.env:
                raise UndefinedName(name)
            value = self.env[name]
        if value is UNKNOWN:
            raise EvalError(f"{name} has no value offline")
        if isinstance(value, _DefaultHome):
            self.diag.counts["kuka_default_home"] += 1
        return value

    def _scope_for(self, name: str) -> dict:
        for frame in reversed(self.locals):
            if name in frame:
                return frame
        return self.env

    # ------------------------------------------------------------------ loading
    def load(self, path: Path) -> None:
        text = read_program_text(path)
        current: Optional[tuple] = None  # (name, params, stmts, modes, is_function)
        for ln, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("&"):
                continue
            line = strip_comment(line, ";").strip().upper()
            if not line:
                continue
            kind, groups = "stmt", ()
            for k, rx in _HEADER_RES:
                m = rx.fullmatch(line)
                if m:
                    kind, groups = k, m.groups()
                    break
            if kind in ("defdat", "enddat"):
                continue
            if kind in ("def", "deffct"):
                name, raw_params = (groups[0], groups[1]) if kind == "def" else (groups[1], groups[2])
                params, modes = [], []
                for item in split_top_level(raw_params, ","):
                    if item.strip():
                        pname, _, mode = item.partition(":")
                        params.append(pname.replace("[]", "").strip())
                        modes.append("IN" if mode.strip() == "IN" else "OUT")  # KRL default is :OUT
                current = (name, params, [], modes, kind == "deffct")
                continue
            if kind in ("end", "endfct"):
                if current:
                    name, params, stmts, modes, is_function = current
                    self.routines[name] = Routine(name, stmts, params, path.name, modes, is_function)
                    self.routine_order.append(name)
                current = None
                continue
            stmt = Stmt(kind, line, ln, path.name, {"groups": groups})
            if current is not None:
                current[2].append(stmt)
            elif kind == "stmt":
                self.execute(stmt)  # data list declarations in .dat files

    # ------------------------------------------------------------------ evaluation
    def eval_text(self, text: str):
        return self.eval(_PARSER.parse(text))

    def eval(self, node):
        if isinstance(node, Token):
            raise EvalError(f"unexpected token {node}")
        d, ch = node.data, node.children
        if d == "number":
            return float(ch[0])
        if d == "string":
            return str(ch[0])[1:-1]
        if d == "enum":
            return "#" + str(ch[0])
        if d == "name":
            return self.lookup(str(ch[0]))
        if d in ("struct", "typed_struct"):
            fields = ch[-1]
            out: dict[str, Any] = {}
            if fields is not None:
                for f in fields.children:
                    out[str(f.children[0])] = self.eval(f.children[-1])
            return out
        if d in ("add", "sub", "mul", "div"):
            a, b = self.eval(ch[0]), self.eval(ch[1])
            return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b}[d]()
        if d in ("eq", "ne"):
            a, b = self.eval(ch[0]), self.eval(ch[1])
            same = abs(a - b) < 1e-9 if _is_number(a) and _is_number(b) else a == b
            return same if d == "eq" else not same
        if d in ("lt", "gt", "le", "ge"):
            a, b = self.eval(ch[0]), self.eval(ch[1])
            if not (_is_number(a) and _is_number(b)):
                raise EvalError("comparison needs numbers")
            return {"lt": a < b, "gt": a > b, "le": a <= b, "ge": a >= b}[d]
        if d in ("and_", "or_", "xor"):
            a, b = self.eval(ch[0]), self.eval(ch[1])
            if not (isinstance(a, bool) and isinstance(b, bool)):
                raise EvalError("AND/OR/EXOR need BOOL operands (put comparisons in parentheses)")
            return {"and_": a and b, "or_": a or b, "xor": a != b}[d]
        if d == "not_":
            v = self.eval(ch[0])
            if not isinstance(v, bool):
                raise EvalError("NOT needs a BOOL operand")
            return not v
        if d == "neg":
            return -self.eval(ch[0])
        if d == "geo":
            left, right = self.eval(ch[0]), self.eval(ch[1])
            out = T_to_frame(frame_to_T(left) @ frame_to_T(right))
            for k in ("S", "T"):
                if isinstance(right, dict) and k in right:
                    out[k] = right[k]
            return out
        if d == "field":
            base = self.eval(ch[0])
            key = str(ch[1])
            if not isinstance(base, dict) or key not in base:
                raise EvalError(f"no component {key}")
            return base[key]
        if d == "index":
            idx = tuple(int(self.eval(c)) for c in ch[1].children)
            if ch[0].data == "name" and str(ch[0].children[0]) in ("TOOL_DATA", "BASE_DATA") and len(idx) == 1:
                return self._array_frame(str(ch[0].children[0]), idx[0])
            base = self.eval(ch[0])
            if not isinstance(base, KArray) or idx not in base:
                raise EvalError(f"array element {idx} not defined")
            return base[idx]
        if d == "call":
            name, arg_nodes = str(ch[0]), ch[1].children
            routine = self.routines.get(name)
            if routine is not None:  # user DEFFCT (or DEF) defined in a loaded file
                args = [self._eval_or_unknown(a) for a in arg_nodes]
                lvalues = [a if _is_lvalue(a) else None for a in arg_nodes]
                result = self._invoke(routine, args, lvalues)
                if result is _NO_RETURN or result is UNKNOWN:
                    raise EvalError(f"{name}() returned no value")
                return result
            args = [self.eval(a) if a is not None else None for a in arg_nodes]
            return self._function(name, args)
        raise EvalError(f"unsupported expression {d}")

    def _array_frame(self, table: str, n) -> dict:
        n = int(n)
        if n == 0:
            return dict(self.env["$NULLFRAME"])
        arr = self.env[table]
        if (n,) not in arr:
            kind = "tools" if table == "TOOL_DATA" else "bases"
            self.diag.need("tool" if table == "TOOL_DATA" else "base", n, self.where)
            self.diag.warn(
                f"{table}[{n}] is not defined (pass $config.dat via kuka.include_dat or set '{kind}: {{{n}: ...}}' "
                "in the config); using identity",
                key=f"missing:{table}:{n}",
            )
            return dict(self.env["$NULLFRAME"])
        return arr[(n,)]

    def _function(self, name: str, args: list):
        def arg(i, default=None):
            return args[i] if i < len(args) and args[i] is not None else default

        def get(struct, key, default=0.0):
            return struct.get(key, default) if isinstance(struct, dict) else default

        if name == "SVEL_JOINT":
            return float(arg(0, 100.0))
        if name == "SACC_JOINT":
            return float(get(arg(0), "ACC", 100.0))
        if name == "SVEL_CP":
            vel = dict(self.env["$VEL"])
            vel["CP"] = float(arg(0, vel["CP"]))
            return vel
        if name == "SACC_CP":
            acc = dict(self.env["$ACC"])
            acc["CP"] = self.cfg["motion"]["cart_accel"] / 1000.0 * float(get(arg(0), "ACC", 100.0)) / 100.0
            return acc
        if name in ("SAPO", "SAPO_PTP"):
            dist = float(get(arg(0), "APO_DIST", 0.0))
            apo = dict(self.env["$APO"])
            apo.update(CDIS=dist, CPTP=dist)
            return apo
        if name == "STOOL2":
            return self._array_frame("TOOL_DATA", get(arg(0), "TOOL_NO", 0))
        if name == "SBASE":
            return self._array_frame("BASE_DATA", arg(0, 0))
        math_fns = {
            "ABS": abs, "SQRT": math.sqrt,
            "SIN": lambda a: math.sin(math.radians(a)), "COS": lambda a: math.cos(math.radians(a)),
            "TAN": lambda a: math.tan(math.radians(a)), "ACOS": lambda a: math.degrees(math.acos(a)),
            "ATAN2": lambda y, x: math.degrees(math.atan2(y, x)),
        }
        if name in math_fns:
            return float(math_fns[name](*[float(a) for a in args]))
        if name == "INV_POS":
            return T_to_frame(np.linalg.inv(frame_to_T(arg(0))))
        if name in ("SIPO_MODE", "SLOAD", "SGEAR_JERK", "SJERK", "USE_CM_PRO_VALUES", "SORI_TYP", "SCIRC_TYP",
                    "SCIRC_MODE", "SPATH_JERK"):
            return None
        raise EvalError(f"unsupported function {name}")

    # ------------------------------------------------------------------ execution
    def for_values(self, s: Stmt):
        var, a, b, step = s.data["groups"]
        try:
            return frange(self.eval_text(a), self.eval_text(b), self.eval_text(step) if step else 1.0)
        except (EvalError, LarkError, TypeError):
            return None

    def set_loop_var(self, s: Stmt, value: float) -> None:
        self._scope_for(s.data["groups"][0])[s.data["groups"][0]] = float(value)

    def eval_condition(self, s: Stmt) -> Optional[bool]:
        groups = s.data.get("groups") or ()
        if not groups or not groups[0]:
            return None
        try:
            value = self.eval_text(groups[0])
        except EVAL_ERRORS:
            return None
        return value if isinstance(value, bool) else None

    def eval_switch(self, switch: Stmt, cases: list[Stmt]) -> Optional[int]:
        try:
            value = self.eval_text(switch.data["groups"][0])
            default = None
            for i, case in enumerate(cases):
                if case.kind == "default":
                    default = i
                    continue
                for item in split_top_level(case.data["groups"][0], ","):
                    v = self.eval_text(item)
                    if (abs(v - value) < 1e-9) if _is_number(v) and _is_number(value) else v == value:
                        return i
        except EVAL_ERRORS:
            return None
        return -1 if default is None else default

    def call(self, name: str, args=None, where: str = "") -> bool:
        routine = self.routines.get(name)
        if routine is None:
            return False
        self._invoke(routine, args or [], [])
        return True

    def _invoke(self, routine: Routine, args: list, lvalues: list):
        """Run a DEF/DEFFCT with KRL parameter passing; returns the RETURN value (or _NO_RETURN)."""
        if self.depth >= MAX_CALL_DEPTH:
            raise EvalError(f"call depth limit reached calling {routine.name}")
        frame: dict[str, Any] = {}
        for i, (pname, mode) in enumerate(zip(routine.params, routine.modes)):
            val = args[i] if i < len(args) and args[i] is not None else UNKNOWN
            frame[pname] = copy.deepcopy(val) if mode == "IN" else val
        self.locals.append(frame)
        self.depth += 1
        result = _NO_RETURN
        try:
            self.run(routine, 0, len(routine.stmts))
        except ReturnSignal as ret:
            result = ret.value
        finally:
            self.depth -= 1
            self.locals.pop()
        for i, (pname, mode) in enumerate(zip(routine.params, routine.modes)):  # :OUT copies back to the caller
            if mode == "OUT" and i < len(lvalues) and lvalues[i] is not None and frame[pname] is not UNKNOWN:
                self._assign_tree(lvalues[i], frame[pname])
        return result

    def _eval_or_unknown(self, node):
        if node is None:
            return UNKNOWN
        try:
            return self.eval(node)
        except EVAL_ERRORS as exc:
            self._note_undefined(exc)
            return UNKNOWN

    def _note_undefined(self, exc) -> None:
        if isinstance(exc, UndefinedName) and not exc.name.startswith("$"):
            self.diag.need("name", exc.name, self.where)

    def run_program(self, entry: Optional[str], main_stem: str) -> None:
        name = (entry or "").upper() or None
        if name is None:
            stem = main_stem.upper()
            procs = [r for r in self.routine_order if not self.routines[r].is_function]
            name = stem if stem in self.routines else (procs[0] if procs else None)
        if name is None or name not in self.routines:
            if not self.routines:
                raise RobotParseError("no DEF ... END routine found: is this a KUKA .src program?")
            raise RobotParseError(f"routine '{entry or main_stem}' not found; routines in the loaded files: "
                                  + ", ".join(self.routine_order))
        self.call(name)

    def execute(self, s: Stmt) -> None:
        where = f"{s.file}:{s.line}"
        self.where = where
        text = s.text
        try:
            m = DECL_RE.match(text) or TYPED_DECL_RE.match(text)
            if m:
                self._declare(m.group(1), m.group(2))
                return
            m = RETURN_RE.match(text)
            if m:
                value = None
                if m.group(1):
                    try:
                        value = self.eval_text(m.group(1))
                    except EVAL_ERRORS as exc:
                        self.diag.warn(f"{where}: RETURN value cannot be computed offline: {str(exc).splitlines()[0]}")
                        value = UNKNOWN
                raise ReturnSignal(value)
            m = MOTION_RE.match(text)
            if m:
                self.diag.motion_statements += 1
                try:
                    self._motion(m.group(1), m.group(2), s)
                except EVAL_ERRORS as exc:
                    self._note_undefined(exc)
                    reason = ("uses data that is not defined in the loaded files" if isinstance(exc, UndefinedName)
                              else str(exc).splitlines()[0])
                    self.diag.drop(where, reason)
                return
            if text in ("SPLINE", "PTP_SPLINE") or text.startswith(("SPLINE ", "PTP_SPLINE ")):
                self.spline_block = len(self.commands)
                w = WITH_RE.split(text, maxsplit=1)
                if len(w) == 2:
                    self._with(w[1])
                return
            if text.startswith("ENDSPLINE"):
                self._end_spline(text, s)
                return
            m = re.match(r"^WAIT\s+SEC\s+(.+)$", text)
            if m:
                self.commands.append(Dwell(float(self.eval_text(m.group(1))), Source(s.file, s.line, text)))
                return
            m = ASSIGN_RE.match(text)
            if m:
                self._assign(m.group(1), self.eval_text(m.group(2)))
                return
            m = CALL_RE.match(text)
            if m:
                name = m.group(1)
                arg_texts = split_top_level(m.group(2), ",") if m.group(2).strip() else []
                if name == "BAS":
                    self._bas([self._try_eval(a) for a in arg_texts], where)
                elif name in self.routines:
                    trees = [_PARSER.parse(a) if a else None for a in arg_texts]
                    self._invoke(self.routines[name], [self._eval_or_unknown(t) for t in trees],
                                 [t if _is_lvalue(t) else None for t in trees])
                else:
                    self.diag.need("routine", name, where)
                    self.diag.warn(f"{where}: call to unknown routine {name} ignored", key=f"call:{name}")
                return
            if text.startswith(("WAIT FOR", "HALT", "INTERRUPT", "EXT ", "EXTFCT ", "SIGNAL ", "GLOBAL INTERRUPT", "TRIGGER", "BRAKE", "RESUME",
                                "CONTINUE", "PULSE", "ANOUT", "ANIN")):
                return
            self.diag.warn(f"{where}: statement ignored: {text[:60]}", key=f"stmt:{text.split()[0]}")
        except EVAL_ERRORS as exc:
            self._note_undefined(exc)
            self.diag.warn(f"{where}: could not interpret '{text[:80]}': {str(exc).splitlines()[0]}")

    def _try_eval(self, text: str):
        try:
            return self.eval_text(text) if text else None
        except (EvalError, LarkError):
            return None

    def _declare(self, typ: str, rest: str) -> None:
        for item in split_top_level(rest, ","):
            m = re.match(r"^(\$?\w+)\s*(\[[^\]]*\])?\s*(?:=\s*(.+))?$", item)
            if not m:
                continue
            name, dims, value = m.groups()
            scope = self.locals[-1] if self.locals else self.env
            if dims and value is None:
                if name not in scope or not isinstance(scope[name], KArray):
                    scope[name] = KArray()
            elif value is not None:
                scope[name] = self.eval_text(value)
            elif name not in scope:
                scope[name] = {} if typ not in ("INT", "REAL", "BOOL", "CHAR") else UNKNOWN

    def _assign(self, lhs: str, value) -> None:
        self._assign_tree(_PARSER.parse(lhs), value)

    def _assign_tree(self, tree, value) -> None:
        chain = []
        while tree.data in ("field", "index"):
            chain.append(tree)
            tree = tree.children[0]
        name = str(tree.children[0])
        scope = self._scope_for(name)
        if not chain:
            scope[name] = copy.deepcopy(value)
            return
        chain.reverse()
        if name not in scope:
            scope[name] = KArray() if chain[0].data == "index" else {}
        container = scope[name]
        for i, node in enumerate(chain):
            key = str(node.children[1]) if node.data == "field" else tuple(int(self.eval(c)) for c in node.children[1].children)
            if i == len(chain) - 1:
                if value is not None:
                    container[key] = copy.deepcopy(value)
            else:
                if key not in container:
                    container[key] = KArray() if chain[i + 1].data == "index" else {}
                container = container[key]

    def _with(self, text: str) -> None:
        for part in split_top_level(text, ","):
            m = ASSIGN_RE.match(part)
            if m:
                try:
                    self._assign(m.group(1), self.eval_text(m.group(2)))
                except (EvalError, LarkError, TypeError) as exc:
                    self.diag.warn(f"WITH assignment '{part[:60]}' ignored: {str(exc).splitlines()[0]}", key=f"with:{part}")

    def _bas(self, args: list, where: str) -> None:
        cmd = args[0] if args else None
        val = args[1] if len(args) > 1 else None
        env = self.env
        if cmd == "#INITMOV":
            env["$VEL_AXIS"] = KArray({(i,): 100.0 for i in range(1, 7)})
            env["$ACC_AXIS"] = KArray({(i,): 100.0 for i in range(1, 7)})
            env["$VEL"] = {"CP": 2.0, "ORI1": 200.0, "ORI2": 200.0}
            env["$ACC"]["CP"] = self.cfg["motion"]["cart_accel"] / 1000.0
        elif cmd == "#VEL_PTP" and val is not None:
            env["$VEL_AXIS"] = KArray({(i,): float(val) for i in range(1, 7)})
        elif cmd == "#ACC_PTP" and val is not None:
            env["$ACC_AXIS"] = KArray({(i,): float(val) for i in range(1, 7)})
        elif cmd == "#VEL_CP" and val is not None:
            env["$VEL"]["CP"] = float(val)
        elif cmd == "#ACC_CP" and val is not None:
            env["$ACC"]["CP"] = float(val)
        elif cmd == "#TOOL" and val is not None:
            env["$TOOL"] = self._array_frame("TOOL_DATA", val)
        elif cmd == "#BASE" and val is not None:
            env["$BASE"] = self._array_frame("BASE_DATA", val)
        elif cmd in ("#PTP_PARAMS", "#PTP_DAT", "#CP_PARAMS", "#CP_DAT", "#FRAMES"):
            if cmd in ("#PTP_PARAMS", "#CP_PARAMS", "#FRAMES"):
                fdat = env.get("FDAT_ACT", {})
                env["$TOOL"] = self._array_frame("TOOL_DATA", fdat.get("TOOL_NO", 0))
                env["$BASE"] = self._array_frame("BASE_DATA", fdat.get("BASE_NO", 0))
                if fdat.get("IPO_FRAME") == "#TCP":
                    self.diag.warn(f"{where}: external TCP (IPO_FRAME #TCP) is not supported", key="ipo_tcp")
            if cmd in ("#PTP_PARAMS", "#PTP_DAT"):
                pdat = env.get("PDAT_ACT", {})
                env["$ACC_AXIS"] = KArray({(i,): float(pdat.get("ACC", 100.0)) for i in range(1, 7)})
                dist = float(pdat.get("APO_DIST", 0.0))
                env["PTP_APO_BY_DIST"] = pdat.get("APO_MODE") == "#CDIS"
                if env["PTP_APO_BY_DIST"]:
                    env["$APO"]["CDIS"] = dist
                else:
                    env["$APO"]["CPTP"] = dist
                if cmd == "#PTP_PARAMS" and val is not None:
                    env["$VEL_AXIS"] = KArray({(i,): float(val) for i in range(1, 7)})
            if cmd in ("#CP_PARAMS", "#CP_DAT"):
                ldat = env.get("LDAT_ACT", {})
                env["$ACC"]["CP"] = self.cfg["motion"]["cart_accel"] / 1000.0 * float(ldat.get("ACC", 100.0)) / 100.0
                env["$APO"]["CDIS"] = float(ldat.get("APO_DIST", 0.0))
                if cmd == "#CP_PARAMS" and val is not None:
                    env["$VEL"]["CP"] = float(val)

    def _target(self, value, relative: bool, rel_frame: str):
        if not isinstance(value, dict):
            raise EvalError("motion target is not a position structure")
        keys = set(value)
        if keys & set(AXIS_KEYS) and not keys & {"X", "Y", "Z"}:
            if relative:
                return RelativeTarget(joints=np.array([float(value.get(k, 0.0)) for k in AXIS_KEYS]))
            base = self.last_axes or dict.fromkeys(AXIS_KEYS, 0.0)
            full = {k: float(value.get(k, base.get(k, 0.0))) for k in AXIS_KEYS}
            self.last_axes = full
            return JointTarget(np.array([full[k] for k in AXIS_KEYS]))
        if relative:
            return RelativeTarget(frame_to_T(value), frame="tool" if rel_frame == "#TOOL" else "base")
        missing = [k for k in CART_KEYS if k not in value]
        if missing and self.last_cart is None:
            self.diag.warn(f"position is missing components {missing} and no previous position is known; using 0")
        base = self.last_cart or {}
        full = {k: float(value.get(k, base.get(k, 0.0))) for k in CART_KEYS}
        for k in ("S", "T"):
            if k in value:
                full[k] = value[k]
        self.last_cart = full
        config = {}
        if "T" in full:
            config["kuka_T"] = int(full["T"])
        if "S" in full:
            config["kuka_S"] = int(full["S"])
        return CartTarget(frame_to_T(full), config)

    def _motion(self, keyword: str, rest: str, s: Stmt) -> None:
        src = Source(s.file, s.line, s.text)
        approx, rel_frame = None, "#BASE"
        while True:
            m = APPROX_RE.search(rest)
            if not m:
                break
            if m.group(1).startswith("#"):
                rel_frame = m.group(1)
            else:
                approx = approx or m.group(1)
            rest = rest[: m.start()]
        parts = WITH_RE.split(rest, maxsplit=1)
        rest = parts[0]
        if len(parts) == 2:
            self._with(parts[1])
        relative = keyword.endswith("_REL")
        base_kw = keyword.replace("_REL", "").lstrip("S") if keyword != "SPL" else "SPL"
        kind = {"PTP": "PTP", "LIN": "LIN", "CIRC": "CIRC", "PL": "SPLINE", "SPL": "SPLINE"}[base_kw]
        args = split_top_level(rest.strip(), ",")
        circ_angle = None
        exprs = []
        for a in args:
            m = re.match(r"^CA\s+(.+)$", a)
            if m:
                circ_angle = float(self.eval_text(m.group(1)))
            elif a:
                exprs.append(a)
        if not exprs:
            raise EvalError("motion without target")
        target = self._target(self.eval_text(exprs[-1]), relative, rel_frame)
        via = None
        if kind == "CIRC":
            if len(exprs) < 2:
                raise EvalError("CIRC requires an auxiliary point")
            via = self._target(self.eval_text(exprs[0]), relative, rel_frame)

        env = self.env
        ov = float(env.get("$OV_PRO", 100.0)) / 100.0
        vel_axis = min(float(v) for v in env["$VEL_AXIS"].values()) if env["$VEL_AXIS"] else 100.0
        acc_axis = min(float(v) for v in env["$ACC_AXIS"].values()) if env["$ACC_AXIS"] else 100.0
        if kind == "PTP":
            speed = Speed(joint_pct=vel_axis * ov, accel_pct=acc_axis)
        else:
            vel = env["$VEL"]
            speed = Speed(
                tcp=float(vel.get("CP", 0.25)) * 1000.0 * ov,
                ori=min(float(vel.get("ORI1", 200.0)), float(vel.get("ORI2", 200.0))) * ov,
                cart_accel=float(env["$ACC"].get("CP", 2.5)) * 1000.0,
            )
        apo = env["$APO"]
        if approx is None:
            zone = Zone()
        elif approx == "C_PTP" and not env.get("PTP_APO_BY_DIST") and keyword.startswith("PTP"):
            zone = Zone.blended(pct=float(apo.get("CPTP", 0.0)))
        elif approx == "C_VEL":
            zone = Zone.blended(pct=float(apo.get("CVEL", 0.0)))
        elif approx == "C_ORI":
            zone = Zone.blended(dist=float(apo.get("CDIS", 0.0)) or None, ori=float(apo.get("CORI", 0.0)))
            if zone.dist is None:
                zone.pct = 50.0
        else:  # C_DIS, C_SPL
            zone = Zone.blended(dist=float(apo.get("CDIS", 0.0)))
        if self.spline_block is not None:
            zone = Zone.blended(pct=200.0 * self.cfg["motion"]["kuka_spline_join"])
        tool, base = frame_to_T(env["$TOOL"]), frame_to_T(env["$BASE"])
        self.commands.append(Motion(kind, target, speed, zone, tool, base, src, via=via, circ_angle=circ_angle))

    def _end_spline(self, text: str, s: Stmt) -> None:
        start = self.spline_block
        self.spline_block = None
        if start is None or start >= len(self.commands):
            return
        last = self.commands[-1]
        if isinstance(last, Motion):
            last.zone = Zone.blended(dist=float(self.env["$APO"].get("CDIS", 0.0))) if "C_SPL" in text else Zone()


def parse_krl(paths: list[Path], cfg: dict, diag: Diagnostics) -> Program:
    parser = KrlParser(cfg, diag)
    paths = [Path(p) for p in paths]
    extra = [Path(p) for p in cfg.get("kuka", {}).get("include_dat", [])]
    for dat in extra:
        parser.load(dat)
    # load every .dat first so data lists are defined before code runs
    ordered: list[Path] = []
    missing_dat: list[str] = []
    for p in paths:
        if p.suffix.lower() == ".src":
            dat = next((d for d in (p.with_suffix(".dat"), p.with_suffix(".DAT")) if d.exists()), None)
            if dat is None:
                missing_dat.append(p.with_suffix(".dat").name)
            elif dat not in paths and dat not in ordered:
                ordered.append(dat)
    ordered += [p for p in paths if p.suffix.lower() == ".dat"]
    ordered += [p for p in paths if p.suffix.lower() != ".dat"]
    for p in ordered:
        parser.load(p)
    parser.apply_config_frames()
    main = next((p for p in paths if p.suffix.lower() != ".dat"), paths[0])
    parser.run_program(cfg.get("entry"), main.stem)
    if missing_dat and any(kind == "name" for kind, _ in diag.needs):
        for name in missing_dat:  # undefined positions almost always mean the .dat was not uploaded
            diag.need("file", name, "positions and motion data (E6POS, PDAT, LDAT, FDAT) are declared there")
    return Program("kuka", main.stem, parser.commands, list(diag.messages))
