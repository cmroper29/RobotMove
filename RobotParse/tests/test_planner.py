import numpy as np
import pytest

from robotparse import load_config, parse_program
from robotparse.kinematics import RobotModel
from robotparse.output import OutputOptions, to_table
from robotparse.planner import plan
from robotparse.transforms import T_inv

from .conftest import URDF

HEADER = "MODULE T\n    PROC main()\n"
FOOTER = "    ENDPROC\nENDMODULE\n"
DOWN = "[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]"


def rt(x, y, z, q=None):
    return f"[[{x},{y},{z}],{q + ',[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]' if q else DOWN}]"


def rapid(write, body):
    return parse_program([write("t.mod", HEADER + body + FOOTER)], load_config())


def speed(res):
    return np.linalg.norm(np.gradient(res.position, res.t, axis=0), axis=1)


def test_straight_line_duration_matches_trapezoid(write):
    prog = rapid(write, f"MoveL {rt(0,0,0)}, v500, fine, tool0;\nMoveL {rt(1000,0,0)}, v500, fine, tool0;\n")
    cfg = load_config()
    res = plan(prog, cfg, "dumb", dt=0.001)
    a = cfg["motion"]["cart_accel"]
    assert res.t[-1] == pytest.approx(1000 / 500 + 500 / a, abs=0.005)
    assert np.max(speed(res)) == pytest.approx(500, rel=0.01)
    assert np.allclose(res.position[-1], [1000, 0, 0])


def test_fixed_timestep_and_unit_direction(write):
    prog = rapid(write, f"MoveL {rt(0,0,0)}, v100, fine, tool0;\nMoveL {rt(100,0,0)}, v100, fine, tool0;\n")
    res = plan(prog, load_config(), "dumb", dt=0.004)
    assert np.allclose(np.diff(res.t), 0.004)
    cols, table, _ = to_table(res, load_config(), "abb", OutputOptions(direction_axis="z"))
    assert cols[:7] == ["t", "x_mm", "y_mm", "z_mm", "dir_x", "dir_y", "dir_z"]
    assert np.allclose(table[:, 4:7], [0, 0, -1])


def test_zone_rounds_corner_and_fine_stops(write):
    body = (f"MoveL {rt(0,0,0)}, v500, fine, tool0;\nMoveL {rt(300,0,0)}, v500, z50, tool0;\n"
            f"MoveL {rt(300,300,0)}, v500, fine, tool0;\n")
    res = plan(rapid(write, body), load_config(), "dumb", dt=0.002)
    gap = np.min(np.linalg.norm(res.position - [300, 0, 0], axis=1))
    assert 5 < gap < 50  # corner is cut but stays within the zone
    assert np.min(speed(res)[len(res.t) // 4: 3 * len(res.t) // 4]) > 50  # no stop at the corner

    body_fine = body.replace("z50", "fine")
    res2 = plan(rapid(write, body_fine), load_config(), "dumb", dt=0.002)
    assert np.min(np.linalg.norm(res2.position - [300, 0, 0], axis=1)) < 1e-6
    assert res2.t[-1] > res.t[-1]


def test_dwell_holds_position(write):
    body = f"MoveL {rt(0,0,0)}, v100, fine, tool0;\nMoveL {rt(100,0,0)}, v100, fine, tool0;\nWaitTime 1.5;\n"
    res = plan(rapid(write, body), load_config(), "dumb", dt=0.01)
    held = res.t > res.t[-1] - 1.4
    assert np.allclose(res.position[held], [100, 0, 0])


def test_orientation_only_move(write):
    body = (f"MoveL {rt(0,0,0,'[1,0,0,0]')}, v100, fine, tool0;\n"
            f"MoveL {rt(0,0,0,'[0.7071068,0,0,0.7071068]')}, [100,45,5000,1000], fine, tool0;\n")
    res = plan(rapid(write, body), load_config(), "dumb", dt=0.01)
    assert res.t[-1] == pytest.approx(2.0, abs=0.2)
    assert np.allclose(res.position, 0)


@pytest.fixture(scope="module")
def robot():
    cfg = load_config(overrides={"robot": {"urdf": str(URDF), "tip_link": "tool0"}})
    return RobotModel(cfg["robot"]), cfg


def test_fk_ik_roundtrip(robot):
    rob, _ = robot
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(rob.lower * 0.6, rob.upper * 0.6)
        T = rob.fk(q)
        q2, ok = rob.ik(T, q + 0.05)
        assert ok and np.allclose(rob.fk(q2), T, atol=1e-6)


def test_ik_best_respects_configuration_hints(robot):
    rob, _ = robot
    q = rob.to_urdf([10, -60, 100, 40, 50, -30])
    T = rob.fk(q)
    # KUKA turn bits: bit i set -> axis i+1 negative. Ask for the wrist-flipped solution (A4<0, A5<0, A6>0)
    bits = (1 << 1) | (1 << 3) | (1 << 4)
    q_flip, ok, mismatch = rob.ik_best(T, q, {"kuka_T": bits})
    c = rob.to_controller(q_flip)
    assert ok and mismatch == 0 and c[3] < 0 and c[4] < 0 and c[5] > 0
    assert np.allclose(rob.fk(q_flip), T, atol=1e-6)
    q_same, ok, _ = rob.ik_best(T, q, {})
    assert np.allclose(q_same, q, atol=1e-6)


SMART_BODY = (
    "    MoveAbsJ [[0,-90,90,0,90,0],[9E9,9E9,9E9,9E9,9E9,9E9]], v1000, z50, tool0;\n"
    f"    MoveJ {rt(550,-200,450)}, v1000, z50, tool0;\n"
    f"    MoveL {rt(550,-200,250)}, v500, fine, tool0;\n"
    f"    MoveL {rt(700,-200,250)}, v200, z20, tool0;\n"
    f"    MoveL {rt(700,100,250)}, v200, fine, tool0;\n"
)


def test_smart_mode_follows_linear_path_and_limits(write, robot):
    rob, cfg = robot
    prog = rapid(write, SMART_BODY)
    res = plan(prog, cfg, "smart", dt=0.002, robot=rob)
    assert res.joints is not None and res.joints.shape == (len(res.t), 6)
    # TCP equals FK of the reported joints
    k = len(res.t) // 2
    assert np.allclose(rob.fk(rob.to_urdf(res.joints[k]))[:3, 3], res.position[k], atol=1e-6)
    # samples on the straight y = -200 weld lie on the programmed line
    on_line = (res.position[:, 0] > 560) & (res.position[:, 0] < 670) & (np.abs(res.position[:, 1] + 200) < 1)
    assert on_line.sum() > 50
    assert np.allclose(res.position[on_line, 2], 250, atol=0.05)
    assert np.allclose(res.position[on_line, 1], -200, atol=0.05)
    # joint velocity and acceleration limits
    qd = np.diff(np.radians(res.joints), axis=0) / 0.002
    assert np.all(np.max(np.abs(qd), axis=0) <= rob.vel_limit * 1.01)
    qdd = np.diff(qd, axis=0) / 0.002
    assert np.all(np.max(np.abs(qdd), axis=0) <= rob.acc_limit * 1.05)
    # the programmed weld speed is respected
    assert np.max(speed(res)[on_line]) <= 200 * 1.01


def test_smart_joint_move_is_synchronised(write, robot):
    rob, cfg = robot
    body = ("    MoveAbsJ [[0,-90,90,0,90,0],[9E9,9E9,9E9,9E9,9E9,9E9]], v1000, fine, tool0;\n"
            "    MoveAbsJ [[40,-70,80,20,60,90],[9E9,9E9,9E9,9E9,9E9,9E9]], v1000, fine, tool0;\n")
    res = plan(rapid(write, body), cfg, "smart", dt=0.01, robot=rob)
    q0, q1 = np.array([0, -90, 90, 0, 90, 0]), np.array([40, -70, 80, 20, 60, 90])
    frac = (res.joints - q0) / (q1 - q0)
    assert np.allclose(frac, frac[:, :1], atol=1e-6)  # all joints at the same fraction of their move
    assert np.allclose(res.joints[-1], q1, atol=1e-6)


def test_dumb_mode_joint_target_uses_urdf_fk(write, robot):
    rob, cfg = robot
    body = ("    MoveAbsJ [[0,-90,90,0,90,0],[9E9,9E9,9E9,9E9,9E9,9E9]], v1000, fine, tool0;\n"
            f"    MoveL {rt(500,0,500)}, v500, fine, tool0;\n")
    res = plan(rapid(write, body), cfg, "dumb", dt=0.01, robot=rob)
    home = rob.fk(rob.to_urdf([0, -90, 90, 0, 90, 0]))
    assert np.allclose(res.position[0], home[:3, 3], atol=1e-6)
    assert np.allclose(res.position[-1], [500, 0, 500])


def test_output_tool_reports_fixed_tcp(write):
    body = ("    VAR tooldata t1 := [TRUE,[[0,0,100],[1,0,0,0]],[1,[0,0,1],[1,0,0,0],0,0,0]];\n"
            f"    MoveL {rt(0,0,0)}, v100, fine, tool0;\n"
            f"    MoveL {rt(100,0,0)}, v100, fine, t1;\n")
    prog = rapid(write, body)
    res = plan(prog, load_config(overrides={"output_tool": "flange"}), "dumb", dt=0.01)
    # tool0 points down; the flange sits exactly at the programmed tool0 poses. The second move is
    # programmed for t1, so the flange ends 100 mm above that target.
    assert np.allclose(res.position[0], [0, 0, 0])
    assert np.allclose(res.position[-1], [100, 0, 100])
