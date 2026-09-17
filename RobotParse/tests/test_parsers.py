import numpy as np
import pytest

from robotparse import load_config, parse_program
from robotparse.model import CartTarget, Dwell, JointTarget, RelativeTarget
from robotparse.transforms import xyzabc_from_T

from .conftest import EXAMPLES


def pos(m):
    return (m.base @ m.target.T)[:3, 3]


# ----------------------------------------------------------------------------- KUKA
def test_kuka_example_inline_forms_and_expert_code():
    cfg = load_config(EXAMPLES / "kuka" / "config.yaml")
    prog = parse_program([EXAMPLES / "kuka" / "weld_demo.src"], cfg)
    ms = prog.motions
    assert prog.vendor == "kuka" and len(ms) == 16
    assert isinstance(ms[0].target, JointTarget) and np.allclose(ms[0].target.q, [0, -90, 90, 0, 0, 0])
    # ILF PTP with PDAT APO_MODE #CDIS and C_PTP -> distance blend
    assert ms[1].kind == "PTP" and ms[1].zone.dist == 50 and ms[1].speed.joint_pct == 100
    assert np.allclose(ms[1].tool[:3, 3], [150, 0, 0])  # TOOL_DATA[1] from the config
    # ILF LIN via BAS(#CP_PARAMS, 0.5): no CONT -> exact stop
    assert ms[2].kind == "LIN" and ms[2].speed.tcp == pytest.approx(500) and ms[2].zone.fine
    # LDAT APO_DIST 20 and C_DIS
    assert ms[3].zone.dist == 20 and ms[3].speed.tcp == pytest.approx(150)
    # expert CIRC with $APO.CDIS
    assert ms[5].kind == "CIRC" and ms[5].zone.dist == 10 and np.allclose(pos(ms[5]), [550, 0, 250])
    assert np.allclose((ms[5].base @ ms[5].via.T)[:3, 3], [625, 75, 250])
    assert isinstance(ms[6].target, RelativeTarget) and np.allclose(ms[6].target.delta[:3, 3], [0, 0, 100])
    # FOR loop unrolled, partial positions inherit orientation from the previous point
    ys = [pos(m)[1] for m in ms[7:10]]
    assert np.allclose(ys, [-75, 0, 75])
    assert np.allclose(xyzabc_from_T(ms[7].target.T)[3:], [0, 90, 0])
    assert isinstance(prog.commands[11], Dwell) and prog.commands[11].duration == 0.5
    # SPTP with WITH clause and C_SPL
    assert ms[11].speed.joint_pct == 50 and ms[11].zone.dist == 50
    # SPLINE block
    assert [m.kind for m in ms[12:15]] == ["SPLINE"] * 3
    assert ms[12].speed.tcp == pytest.approx(300) and ms[14].zone.fine and not ms[12].zone.fine


def test_krl_expressions(write):
    src = write("expr.src", """DEF expr()
DECL E6POS P
DECL FRAME F
P = {X 100, Y 0, Z 500, A 0, B 90, C 0}
F = {X 10, Y 20, Z 30, A 0, B 0, C 0}
$BASE = {X 1000, Y 0, Z 0, A 90, B 0, C 0}
$VEL.CP = 0.1*2
LIN P:F C_VEL
PTP {A1 10, A2 -80} C_PTP
LIN_REL {X 5} #TOOL
END
""")
    prog = parse_program([src], load_config())
    lin, ptp, rel = prog.motions
    assert lin.speed.tcp == pytest.approx(200)
    assert np.allclose(lin.base[:3, 3], [1000, 0, 0])
    # P:F composes frames: offset along P's axes (B 90 -> P's X is world -Z)
    assert np.allclose(lin.target.T[:3, 3], [100 + 30, 20, 500 - 10])
    assert not lin.zone.fine
    assert isinstance(ptp.target, JointTarget) and np.allclose(ptp.target.q, [10, -80, 0, 0, 0, 0])
    assert isinstance(rel.target, RelativeTarget) and rel.target.frame == "tool"


def test_krl_subroutine_call_and_missing_tool_warns(write):
    src = write("calls.src", """DEF calls()
BAS(#TOOL, 3)
sub1()
END
DEF sub1()
LIN {X 1, Y 2, Z 3, A 0, B 0, C 0}
END
""")
    prog = parse_program([src], load_config())
    assert len(prog.motions) == 1
    assert any("TOOL_DATA[3]" in w for w in prog.warnings)


# ----------------------------------------------------------------------------- ABB
def test_abb_example():
    cfg = load_config(EXAMPLES / "abb" / "config.yaml")
    prog = parse_program([EXAMPLES / "abb" / "demo.mod"], cfg)
    ms = prog.motions
    assert prog.vendor == "abb" and len(ms) == 13
    assert isinstance(ms[0].target, JointTarget) and ms[0].zone.dist == 50
    assert ms[1].kind == "PTP" and ms[1].speed.tcp == 1000
    assert np.allclose(pos(ms[1]), [550, -200, 450])  # wobj uframe offsets x by 400
    assert np.allclose(ms[1].tool[:3, 3], [0, 0, 150])
    assert ms[2].zone.fine
    assert ms[3].speed.tcp == 150 and ms[3].speed.accel_pct == 50 and ms[3].zone.dist == 20
    assert ms[4].zone.dist == 5  # user zonedata
    assert ms[5].kind == "CIRC" and np.allclose((ms[5].base @ ms[5].via.T)[:3, 3], [625, 75, 250])
    assert ms[6].speed.tcp == 250 and np.allclose(pos(ms[6]), [550, 0, 350])  # Offs + \V
    assert isinstance(prog.commands[7], Dwell)
    assert [pos(m)[1] for m in ms[7:10]] == pytest.approx([-75, 0, 75])  # FOR over path{i}
    reltool = ms[10]
    assert reltool.speed.duration == 2 and np.allclose(pos(reltool), [600, 75, 350])
    assert isinstance(ms[11].target, RelativeTarget)  # Offs(CRobT(), ...) inside called PROC
    assert np.allclose(ms[11].target.delta[:3, 3], [0, 0, 100])
    assert ms[0].target.q.tolist() == ms[12].target.q.tolist()


def test_abb_reltool_rotation_and_field_assignment(write):
    mod = write("t.mod", """MODULE T
    VAR robtarget p := [[100,0,0],[1,0,0,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    PROC main()
        p.trans.z := 50;
        MoveL RelTool(p, 0, 0, 10 \\Rz:=90), v100, z1, tool0;
        MoveL p, v100, fine, tool0 \\WObj:=wobj0;
    ENDPROC
ENDMODULE
""")
    prog = parse_program([mod], load_config())
    a, b = prog.motions
    assert np.allclose(a.target.T[:3, 3], [100, 0, 60])
    assert np.allclose(a.target.T[:3, 0], [0, 1, 0])  # tool X rotated 90 deg about tool Z
    assert np.allclose(b.target.T[:3, 3], [100, 0, 50])
    assert a.zone.dist == 1


# ----------------------------------------------------------------------------- FANUC
def test_fanuc_example():
    cfg = load_config(EXAMPLES / "fanuc" / "config.yaml")
    prog = parse_program([EXAMPLES / "fanuc" / "demo.ls"], cfg)
    ms = prog.motions
    assert prog.vendor == "fanuc" and len(ms) == 14
    assert isinstance(ms[0].target, JointTarget)
    assert ms[1].speed.joint_pct == 50 and ms[1].zone.cnt == 50
    assert np.allclose(pos(ms[1]), [550, -200, 450])  # UFRAME 1
    assert ms[1].target.config == {"fanuc_flip": "N", "fanuc_turns": [0, 0, 0]}
    assert ms[3].zone.cnt == 100 and ms[4].speed.accel_pct == 80
    assert ms[5].kind == "CIRC" and ms[5].zone.cnt == 20
    assert np.allclose(pos(ms[6]), [550, 0, 350])  # Offset,PR[1]
    assert [pos(m)[1] for m in ms[7:10]] == pytest.approx([-75, 0, 75])  # PR[2,2] incremented in FOR
    arcs = ms[10:13]
    assert all(m.kind == "CIRC" and m.arc_ref_excluded for m in arcs)
    assert np.allclose((arcs[0].base @ arcs[0].via.T)[:3, 3], pos(arcs[1]))


def test_fanuc_speed_units_and_call(write):
    sub = write("SUB.ls", """/PROG SUB
/MN
   1:L P[1] 600cm/min CNT R[2] ;
/POS
P[1]{
   GP1:
	UF : 0, UT : 0,
	X = 1.0 mm, Y = 2.0 mm, Z = 3.0 mm, W = 0.0 deg, P = 0.0 deg, R = 0.0 deg
};
/END
""")
    main = write("MAIN.ls", """/PROG MAIN
/MN
   1:  R[2]=40 ;
   2:  CALL SUB ;
   3:J P[1] 2sec FINE ;
   4:  OVERRIDE=50% ;
   5:L P[1] 90deg/sec FINE ;
/POS
P[1]{
   GP1:
	UF : 0, UT : 0,
	J1= 1.0 deg, J2= 2.0 deg, J3= 3.0 deg, J4= 4.0 deg, J5= 5.0 deg, J6= 6.0 deg
};
/END
""")
    prog = parse_program([main, sub], load_config())
    a, b, c = prog.motions
    assert a.speed.tcp == pytest.approx(100) and a.zone.cnt == 40
    assert b.speed.duration == 2 and isinstance(b.target, JointTarget)
    assert c.speed.ori == pytest.approx(45)


# ----------------------------------------------------------------------------- KUKA functions and logic
KRL_FUNCTIONS = """DEF fct( )
DECL E6POS P
DECL INT N, K, TOOLNR
DECL REAL V, R, UNSET
P = {X 500, Y 0, Z 300, A 0, B 90, C 0}
V = WeldSpeed(3)
$VEL.CP = V
LIN LiftZ(P, 50)            ; DEFFCT at the end of the file, P passed :IN
NextRow(P, 25)              ; DEF with an explicit :OUT struct
LIN P
Grow(R)                     ; default parameter mode is :OUT
LIN {X 500, Y R, Z 300}
N = 0
WHILE N < 3
  N = N + 1
  LIN {X 600, Y N*10, Z 300}
ENDWHILE
K = 0
REPEAT
  K = K + 2
UNTIL K >= 4
EarlyOut(1)
IF (N == 3) AND NOT (K > 10) THEN
  LIN Offset(P, 5)          ; GLOBAL DEFFCT from another file
ELSE
  LIN {X 0, Y 0, Z 0}
ENDIF
TOOLNR = 2
SWITCH TOOLNR
CASE 1
  LIN {X 111, Y 0, Z 300}
CASE 2, 3
  LIN {X 222, Y 0, Z 300}
DEFAULT
  LIN {X 333, Y 0, Z 300}
ENDSWITCH
IF $IN[7] THEN
  LIN {X 800, Y 0, Z 300}
ENDIF
$VEL.CP = UNSET
LIN {X 900, Y Broken(1), Z 300}
END

DEFFCT REAL WeldSpeed(LAYER:IN)
  DECL INT LAYER
  IF LAYER > 2 THEN
    RETURN 0.05
  ELSE
    RETURN 0.2
  ENDIF
ENDFCT

DEFFCT E6POS LiftZ(Q:IN, DZ:IN)
  DECL E6POS Q
  DECL REAL DZ
  Q.Z = Q.Z + DZ
  RETURN Q
ENDFCT

DEF NextRow(Q:OUT, DY:IN)
  DECL E6POS Q
  Q.Y = Q.Y + DY
END

DEF Grow(G)
  DECL REAL G
  G = 42
END

DEF EarlyOut(X:IN)
  IF X == 1 THEN
    RETURN
  ENDIF
  LIN {X 0, Y 0, Z 0}
END

DEFFCT REAL Broken(X:IN)
  DECL INT X
ENDFCT
"""

KRL_LIB = """GLOBAL DEFFCT E6POS Offset(P:IN, D:IN)
  DECL E6POS P
  DECL REAL D
  P.Z = P.Z + D * 2
  RETURN P
ENDFCT
"""


def test_krl_functions_parameters_and_logic(write):
    prog = parse_program([write("fct.src", KRL_FUNCTIONS), write("lib.src", KRL_LIB)], load_config())
    pts = [m.target.T[:3, 3] for m in prog.motions]
    expected = [
        [500, 0, 350],   # LiftZ result; P itself unchanged (:IN copy)
        [500, 25, 300],  # NextRow wrote back through :OUT
        [500, 42, 300],  # Grow wrote back a scalar (default :OUT)
        [600, 10, 300], [600, 20, 300], [600, 30, 300],  # WHILE evaluated
        [500, 25, 310],  # IF evaluated, GLOBAL DEFFCT from lib.src; EarlyOut returned before its LIN
        [222, 0, 300],   # SWITCH evaluated
        [800, 0, 300],   # IF on a run-time input: first branch, with a warning
    ]
    assert len(pts) == len(expected)
    for got, want in zip(pts, expected):
        assert np.allclose(got, want)
    assert prog.motions[0].speed.tcp == pytest.approx(50)  # WeldSpeed(3) took the LAYER > 2 branch
    assert prog.motions[-1].speed.tcp == pytest.approx(50)  # $VEL.CP = UNSET did not set 0
    text = "\n".join(prog.warnings)
    assert "$IN[7]" in text and "UNSET has no value" in text and "BROKEN() returned no value" in text
    assert "fct.src:22" not in text  # evaluable conditions produce no warnings
