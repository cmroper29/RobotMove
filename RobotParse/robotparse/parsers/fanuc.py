"""FANUC TP ASCII program parser (.ls)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..model import (CartTarget, Diagnostics, Dwell, JointTarget, Motion, Program, RelativeTarget, RobotParseError,
                     Source, Speed, Zone)
from ..textio import read_program_text
from ..transforms import T_from_xyzwpr, frame_from_spec, xyzwpr_from_T
from .common import BlockRunner, Routine, Stmt, frange

LINE_RE = re.compile(r"^\s*(\d+)\s*:(.*)$")
CONT_RE = re.compile(r"^\s*:(.*)$")
POS_BLOCK_RE = re.compile(r"P\[\s*(\d+)\s*(?::\s*\"[^\"]*\")?\s*\]\s*\{(.*?)\}\s*;", re.S)
MOTION_RE = re.compile(r"^(?P<type>[JLCA])\s+(?P<rest>(?:P|PR)\[.*)$", re.I)
POINT_RE = re.compile(r"^(?P<kind>PR|P)\[\s*(?P<idx>R\[\s*\d+\s*\]|\d+)\s*(?::[^\]]*)?\]\s*(?P<rest>.*)$", re.I)
SPEED_RE = re.compile(
    r"^(?P<val>max_speed|R\[\s*\d+\s*\]|[\d.]+)\s*(?P<unit>%|mm/sec|cm/min|inch/min|deg/sec|msec|sec)?\s*(?P<rest>.*)$", re.I
)
TERM_RE = re.compile(r"^(?P<term>FINE|CNT\s*(?:R\[\s*\d+\s*\]|\d+)|CD\s*\d+|CR\s*\d+)\s*(?P<rest>.*)$", re.I)
REG_RE = re.compile(r"R\[\s*(\d+)\s*(?::[^\]]*)?\]", re.I)
HEADERS = [
    ("if", re.compile(r"IF\s*\(.*\)\s*THEN", re.I)),
    ("else", re.compile(r"ELSE", re.I)),
    ("endif", re.compile(r"ENDIF", re.I)),
    ("for", re.compile(r"FOR\s+R\[\s*(\d+)\s*(?::[^\]]*)?\]\s*=\s*(.+?)\s+(TO|DOWNTO)\s+(.+)", re.I)),
    ("endfor", re.compile(r"ENDFOR", re.I)),
]
MM_PER = {"mm/sec": 1.0, "cm/min": 10.0 / 60.0, "inch/min": 25.4 / 60.0}


class _SkipMotion(Exception):
    """Raised inside motion handling when a motion cannot be built."""


class FanucParser(BlockRunner):
    def __init__(self, cfg: dict, diag: Diagnostics):
        super().__init__(diag)
        self.cfg = cfg
        self.commands: list = []
        self.points: dict[str, dict[int, dict]] = {}
        self.program_order: list[str] = []
        fc = cfg.get("fanuc", {})
        self.registers: dict[int, float] = {int(k): float(v) for k, v in fc.get("registers", {}).items()}
        self.pr: dict[int, dict] = {}
        for n, spec in fc.get("position_registers", {}).items():
            if "joints" in spec:
                self.pr[int(n)] = {"joints": np.asarray(spec["joints"], float)}
            else:
                self.pr[int(n)] = {"xyzwpr": xyzwpr_from_T(frame_from_spec(spec))}
        self.uframe = 0
        self.utool = 0
        self.override = 100.0
        self.offset_pr: Optional[int] = None
        self.tool_offset_pr: Optional[int] = None
        self.current_program = ""
        self.current_where = ""
        self.arc_run: list[Motion] = []

    # ------------------------------------------------------------------ loading
    def load(self, path: Path) -> None:
        text = read_program_text(path)
        name = path.stem.upper()
        seen_mn = False
        section = None
        stmts: list[Stmt] = []
        pos_text: list[str] = []
        pending: Optional[list] = None
        for ln, raw in enumerate(text.splitlines(), 1):
            m = re.match(r"^/(PROG|ATTR|APPL|MN|POS|END)\b\s*(\S*)", raw.strip())
            if m:
                section = m.group(1)
                seen_mn = seen_mn or section == "MN"
                if section == "PROG" and m.group(2):
                    name = m.group(2).upper()
                continue
            if section == "MN":
                lm = LINE_RE.match(raw)
                cm = CONT_RE.match(raw)
                if lm:
                    pending = [int(lm.group(1)), lm.group(2)]
                elif cm and pending is not None:
                    pending[1] += " " + cm.group(1)
                else:
                    continue
                if pending[1].rstrip().endswith(";"):
                    body = pending[1].rstrip()[:-1].strip()
                    stmts.append(self._classify(body, ln, path.name))
                    pending = None
            elif section == "POS":
                pos_text.append(raw)
        if not seen_mn:
            raise RobotParseError(f"{path.name} is not a FANUC ASCII program: it has no /MN section "
                                  "(expected a .ls file with /PROG, /MN and /POS sections)")
        self.points[name] = self._parse_positions("\n".join(pos_text))
        self.routines[name] = Routine(name, stmts, file=path.name)
        self.program_order.append(name)

    @staticmethod
    def _classify(body: str, line: int, file: str) -> Stmt:
        text = re.sub(r"\s+", " ", body).strip()
        for kind, rx in HEADERS:
            m = rx.fullmatch(text)
            if m:
                return Stmt(kind, text, line, file, {"groups": m.groups()})
        return Stmt("stmt", text, line, file)

    @staticmethod
    def _parse_positions(text: str) -> dict[int, dict]:
        points = {}
        for m in POS_BLOCK_RE.finditer(text):
            idx, body = int(m.group(1)), m.group(2)
            gp = re.split(r"GP\d+\s*:", body)
            group = gp[1] if len(gp) > 1 else body
            data: dict = {}
            uf = re.search(r"UF\s*:\s*(\w+)", group)
            ut = re.search(r"UT\s*:\s*(\w+)", group)
            data["uf"] = uf.group(1) if uf else "F"
            data["ut"] = ut.group(1) if ut else "F"
            cfg = re.search(r"CONFIG\s*:\s*'([^']*)'", group)
            vals = dict(re.findall(r"\b([XYZWPR]|J\d+)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", group))
            if "X" in vals:
                data["xyzwpr"] = np.array([float(vals.get(k, 0.0)) for k in "XYZWPR"])
            else:
                joints = sorted((int(k[1:]), float(v)) for k, v in vals.items() if k.startswith("J"))
                data["joints"] = np.array([v for _, v in joints])
            if cfg:
                data["config"] = _parse_config(cfg.group(1))
            points[idx] = data
        return points

    # ------------------------------------------------------------------ helpers
    def _reg(self, n: int, where: str) -> float:
        if n not in self.registers:
            self.diag.need("register", n, where)
            self.diag.warn(f"{where}: register R[{n}] has no value (set fanuc.registers in the config); using 0",
                           key=f"reg:{n}")
        return self.registers.get(n, 0.0)

    def _number(self, text: str, where: str) -> float:
        expr = REG_RE.sub(lambda m: repr(self._reg(int(m.group(1)), where)), text)
        expr = re.sub(r"PR\[\s*(\d+)\s*,\s*(\d+)\s*(?::[^\]]*)?\]",
                      lambda m: repr(self._pr_component(int(m.group(1)), int(m.group(2)), where)), expr, flags=re.I)
        if not re.fullmatch(r"[\d.eE+\-*/() ]+", expr):
            raise ValueError(f"cannot evaluate '{text}'")
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - digits/operators only

    def _pr_component(self, n: int, i: int, where: str) -> float:
        pr = self.pr.get(n)
        if pr is None:
            self.diag.need("position_register", n, where)
            self.diag.warn(f"{where}: PR[{n}] has no value (set fanuc.position_registers in the config)", key=f"pr:{n}")
            return 0.0
        vec = pr.get("xyzwpr", pr.get("joints"))
        return float(vec[i - 1])

    def _frame(self, table: str, n: int) -> np.ndarray:
        if n == 0:
            return np.eye(4)
        spec = self.cfg.get(table, {}).get(n)
        if spec is None:
            label = "UFRAME" if table == "bases" else "UTOOL"
            self.diag.need("base" if table == "bases" else "tool", n, self.current_where)
            self.diag.warn(f"{label} {n} is not defined (set '{table}: {{{n}: ...}}' in the config); using identity",
                           key=f"{table}:{n}")
            return np.eye(4)
        return frame_from_spec(spec)

    def _point(self, kind: str, idx_text: str, where: str) -> Optional[dict]:
        idx = int(self._number(idx_text, where)) if idx_text.upper().startswith("R") else int(idx_text)
        if kind.upper() == "PR":
            pr = self.pr.get(idx)
            if pr is None:
                self.diag.need("position_register", idx, where)
                raise _SkipMotion(f"PR[{idx}] has no value (set fanuc.position_registers in the config)")
            return pr
        p = self.points.get(self.current_program, {}).get(idx)
        if p is None:
            raise _SkipMotion(f"P[{idx}] is not in the /POS section of {self.current_program}")
        return p

    # ------------------------------------------------------------------ execution
    def for_values(self, s: Stmt):
        reg, start, direction, stop = s.data["groups"]
        where = f"{s.file}:{s.line}"
        try:
            a, b = self._number(start, where), self._number(stop, where)
        except ValueError:
            return None
        return frange(a, b, 1.0 if direction.upper() == "TO" else -1.0)

    def set_loop_var(self, s: Stmt, value: float) -> None:
        self.registers[int(s.data["groups"][0])] = value

    def run_program(self, entry: Optional[str]) -> None:
        name = (entry or self.program_order[0]).upper()
        if name not in self.routines:
            raise RobotParseError(f"program '{name}' not found; loaded programs: {', '.join(self.program_order)}")
        self._call_program(name)
        self._flush_arcs()

    def _call_program(self, name: str, where: str = "") -> bool:
        prev = self.current_program
        self.current_program = name
        try:
            return self.call(name, where=where)
        finally:
            self.current_program = prev

    def execute(self, s: Stmt) -> None:
        where = f"{s.file}:{s.line}"
        self.current_where = where
        text = s.text
        try:
            if text.startswith(("!", "//")) or not text:
                return
            m = MOTION_RE.match(text)
            if m:
                self.diag.motion_statements += 1
                try:
                    self._motion(m.group("type").upper(), m.group("rest"), s)
                except (_SkipMotion, ValueError, KeyError, IndexError) as exc:
                    self.diag.drop(where, str(exc))
                return
            self._flush_arcs()
            up = text.upper()
            m = re.fullmatch(r"(UFRAME_NUM|UTOOL_NUM)\s*=\s*(.+)", up)
            if m:
                val = int(self._number(m.group(2), where))
                if m.group(1) == "UFRAME_NUM":
                    self.uframe = val
                else:
                    self.utool = val
                return
            m = re.fullmatch(r"WAIT\s+(.+?)\s*\(SEC\)", up)
            if m:
                self.commands.append(Dwell(self._number(m.group(1), where), Source(s.file, s.line, text)))
                return
            m = re.fullmatch(r"OVERRIDE\s*=\s*(.+?)%", up)
            if m:
                self.override = self._number(m.group(1), where)
                return
            m = re.fullmatch(r"CALL\s+(\w+)(?:\s*\(.*\))?", up)
            if m:
                if not self._call_program(m.group(1), where):
                    self.diag.need("routine", m.group(1), where)
                    self.diag.warn(f"{where}: CALL {m.group(1)}: program not loaded (pass its .ls file too); skipped",
                                   key=f"call:{m.group(1)}")
                return
            m = re.fullmatch(r"(OFFSET|TOOL_OFFSET)\s+CONDITION\s+PR\[\s*(\d+)\s*(?::[^\]]*)?\].*", up)
            if m:
                if m.group(1) == "OFFSET":
                    self.offset_pr = int(m.group(2))
                else:
                    self.tool_offset_pr = int(m.group(2))
                return
            if self._register_assignment(up, where):
                return
            if up.startswith(("JMP", "IF ", "SELECT", "SKIP")):
                self.diag.warn(f"{where}: jumps/conditional branches are not followed: {text[:60]}", key=f"jmp:{where}")
                return
            if up == "END":
                return
        except (ValueError, KeyError, IndexError) as exc:
            self.diag.warn(f"{where}: could not interpret '{text[:80]}': {exc}")

    def _register_assignment(self, up: str, where: str) -> bool:
        m = re.fullmatch(r"R\[\s*(\d+)\s*(?::[^\]]*)?\]\s*=\s*(.+)", up)
        if m:
            self.registers[int(m.group(1))] = self._number(m.group(2), where)
            return True
        m = re.fullmatch(r"PR\[\s*(\d+)\s*,\s*(\d+)\s*(?::[^\]]*)?\]\s*=\s*(.+)", up)
        if m:
            n, i = int(m.group(1)), int(m.group(2))
            value = self._number(m.group(3), where)
            pr = self.pr.setdefault(n, {"xyzwpr": np.zeros(6)})
            vec = pr.get("xyzwpr", pr.get("joints"))
            vec[i - 1] = value
            return True
        m = re.fullmatch(r"PR\[\s*(\d+)\s*(?::[^\]]*)?\]\s*=\s*(.+)", up)
        if m:
            n, rhs = int(m.group(1)), m.group(2).strip()
            src = POINT_RE.match(rhs)
            if src and not src.group("rest"):
                try:
                    p = self._point(src.group("kind"), src.group("idx"), where)
                    self.pr[n] = {k: np.array(v, float) for k, v in p.items() if k in ("xyzwpr", "joints")}
                except _SkipMotion as exc:
                    self.diag.warn(f"{where}: PR[{n}] not assigned: {exc}")
            elif rhs in ("LPOS", "JPOS"):
                self.diag.counts["runtime_condition"] += 1
                self.diag.warn(f"{where}: PR[{n}]={rhs} uses the live robot position; PR left unchanged")
            elif rhs == "0":
                self.pr[n] = {"xyzwpr": np.zeros(6)}
            return True
        return False

    def _motion(self, mtype: str, rest: str, s: Stmt) -> None:
        where = f"{s.file}:{s.line}"
        src = Source(s.file, s.line, s.text)
        pm = POINT_RE.match(rest)
        if not pm:
            raise _SkipMotion(f"unrecognised motion '{s.text[:60]}'")
        points = [(pm.group("kind"), pm.group("idx"))]
        rest = pm.group("rest")
        if mtype == "C":
            pm2 = POINT_RE.match(rest)
            if not pm2:
                raise _SkipMotion("circular motion without end point")
            points.append((pm2.group("kind"), pm2.group("idx")))
            rest = pm2.group("rest")
        sm = SPEED_RE.match(rest)
        if not sm:
            raise _SkipMotion(f"could not read the speed in '{s.text[:60]}'")
        speed = self._speed(mtype, sm.group("val"), (sm.group("unit") or "").lower(), where)
        rest = sm.group("rest")
        tm = TERM_RE.match(rest)
        zone = Zone()
        if tm:
            term = re.sub(r"\s+", "", tm.group("term").upper())
            rest = tm.group("rest")
            if term.startswith("CNT"):
                val = term[3:]
                cnt = self._number(val, where)
                zone = Zone() if cnt <= 0 else Zone.blended(cnt=cnt)
            elif term.startswith(("CD", "CR")):
                zone = Zone.blended(dist=float(term[2:]))
        options = rest.upper()
        acc = re.search(r"\bACC\s*(\d+)", options)
        if acc:
            speed.accel_pct = float(acc.group(1))
        offset_pr = self._option_pr(options, r"(?<!TOOL_)OFFSET", self.offset_pr)
        tool_offset_pr = self._option_pr(options, r"TOOL_OFFSET", self.tool_offset_pr)
        incremental = bool(re.search(r"\bINC\b", options))

        resolved = []
        for kind, idx in points:
            p = self._point(kind, idx, where)
            resolved.append(self._target(p, incremental, offset_pr, tool_offset_pr, where))
        uf, ut = self._frames_for(points[-1], where)
        motion = Motion(
            {"J": "PTP", "L": "LIN", "C": "CIRC", "A": "CIRC"}[mtype],
            resolved[-1], speed, zone, self._frame("tools", ut), self._frame("bases", uf), src,
            via=resolved[0] if mtype == "C" else None,
        )
        if mtype == "A":
            self.arc_run.append(motion)
        else:
            self._flush_arcs()
        self.commands.append(motion)

    def _option_pr(self, options: str, keyword: str, default: Optional[int]) -> Optional[int]:
        m = re.search(r"\b" + keyword + r"\b\s*(?:,\s*PR\[\s*(\d+)\s*(?::[^\]]*)?\])?", options)
        if not m:
            return None
        return int(m.group(1)) if m.group(1) else default

    def _frames_for(self, point: tuple[str, str], where: str) -> tuple[int, int]:
        uf, ut = self.uframe, self.utool
        if point[0].upper() == "P":
            try:
                p = self.points.get(self.current_program, {}).get(int(point[1]), {})
            except ValueError:
                p = {}
            if p.get("uf", "F") not in ("F", "*"):
                uf = int(p["uf"])
            if p.get("ut", "F") not in ("F", "*"):
                ut = int(p["ut"])
        return uf, ut

    def _speed(self, mtype: str, val: str, unit: str, where: str) -> Speed:
        m = self.cfg["motion"]
        ov = self.override / 100.0
        if val.lower() == "max_speed":
            return Speed(joint_pct=100.0 * ov) if mtype == "J" else Speed(tcp=m["ptp_tcp_speed"] * ov, ori=m["max_ori_speed"])
        v = self._number(val, where)
        if unit == "%":
            return Speed(joint_pct=v * ov)
        if unit in MM_PER:
            return Speed(tcp=v * MM_PER[unit] * ov, ori=m["max_ori_speed"] * ov)
        if unit == "deg/sec":
            return Speed(tcp=m["ptp_tcp_speed"] * ov, ori=v * ov)
        if unit in ("sec", "msec"):
            return Speed(duration=v / (1000.0 if unit == "msec" else 1.0) / ov)
        self.diag.warn(f"{where}: unknown speed unit '{unit}'; using default speed")
        return Speed(tcp=m["default_tcp_speed"] * ov, ori=m["default_ori_speed"])

    def _target(self, p: dict, incremental: bool, offset_pr, tool_offset_pr, where: str):
        if "joints" in p:
            q = np.array(p["joints"], float)
            if incremental:
                return RelativeTarget(joints=q)
            if offset_pr is not None or tool_offset_pr is not None:
                self.diag.warn(f"{where}: offsets on joint positions are not supported; offset ignored")
            return JointTarget(q)
        T = T_from_xyzwpr(*p["xyzwpr"])
        if incremental:
            return RelativeTarget(T, frame="base")
        if offset_pr is not None:
            T = self._pr_frame(offset_pr, where) @ T
        if tool_offset_pr is not None:
            T = T @ self._pr_frame(tool_offset_pr, where)
        config = {}
        if "config" in p:
            config = p["config"]
        return CartTarget(T, config)

    def _pr_frame(self, n: int, where: str) -> np.ndarray:
        pr = self.pr.get(n)
        if pr is None or "xyzwpr" not in pr:
            if pr is None:
                self.diag.need("position_register", n, where)
            self.diag.warn(f"{where}: offset PR[{n}] is unknown or not Cartesian; offset ignored", key=f"prf:{n}")
            return np.eye(4)
        return T_from_xyzwpr(*pr["xyzwpr"])

    def _flush_arcs(self) -> None:
        """Convert a run of FANUC 'A' moves: each arc passes through its neighbours."""
        run = self.arc_run
        self.arc_run = []
        if not run:
            return
        if len(run) < 3:
            for mo in run:
                mo.kind = "LIN"
            if len(run) == 2:
                self.diag.warn(f"{run[0].source}: circular arc (A) needs 3+ points; treated as linear")
            return
        for i, mo in enumerate(run):
            if i < len(run) - 1:
                mo.via, mo.arc_ref_excluded = run[i + 1].target, True
            else:
                mo.via, mo.arc_ref_excluded = run[i - 2].target, True


def _parse_config(text: str) -> dict:
    parts = [p.strip() for p in text.split(",")]
    letters = parts[0].split()
    out: dict = {}
    if letters:
        out["fanuc_flip"] = letters[0].upper()
    turns = []
    for p in parts[1:4]:
        try:
            turns.append(int(p))
        except ValueError:
            pass
    if len(turns) == 3:
        out["fanuc_turns"] = turns
    return out


def parse_ls(paths: list[Path], cfg: dict, diag: Diagnostics) -> Program:
    parser = FanucParser(cfg, diag)
    for p in paths:
        parser.load(Path(p))
    parser.run_program(cfg.get("entry"))
    return Program("fanuc", parser.program_order[0], parser.commands, list(diag.messages))
