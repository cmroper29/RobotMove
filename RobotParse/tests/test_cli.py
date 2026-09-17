import csv

import pytest

from robotparse.cli import main

from .conftest import EXAMPLES


@pytest.mark.parametrize("vendor,program", [("kuka", "weld_demo.src"), ("abb", "demo.mod"), ("fanuc", "demo.ls")])
@pytest.mark.parametrize("mode", ["dumb", "smart"])
def test_examples_run_end_to_end(tmp_path, vendor, program, mode):
    out = tmp_path / f"{vendor}_{mode}.csv"
    cfg = EXAMPLES / vendor / "config.yaml"
    rc = main([str(EXAMPLES / vendor / program), "-c", str(cfg), "-m", mode, "--dt", "0.02", "-o", str(out),
               "--quat", "--speed", "--joints", "--source", "--units", "m", "-q"])
    assert rc == 0
    with open(out) as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert header[:7] == ["t", "x_m", "y_m", "z_m", "dir_x", "dir_y", "dir_z"]
    assert "qw" in header and "speed_m_s" in header and header[-1] == "source"
    assert ("joint_a1" in header) == (mode == "smart")
    assert len(rows) > 100
    t = [float(r[0]) for r in rows[1:]]
    assert all(abs((b - a) - 0.02) < 1e-9 for a, b in zip(t, t[1:]))
    speeds = [float(r[header.index("speed_m_s")]) for r in rows[1:]]
    assert max(speeds) < 3.0  # no TCP jumps: example configs set output_tool


def test_list_mode(capsys):
    assert main([str(EXAMPLES / "fanuc" / "demo.ls"), "-c", str(EXAMPLES / "fanuc" / "config.yaml"), "--list"]) == 0
    assert "14 motions" in capsys.readouterr().out


def test_strict_fails_on_warnings(tmp_path):
    prog = tmp_path / "w.mod"
    prog.write_text("MODULE W\nPROC main()\nMoveL [[500,0,300],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]], v100, fine, tool0;\n"
                    "IF DInput(diStart) = 1 THEN\n"
                    "MoveL [[600,0,300],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]], v100, fine, tool0;\n"
                    "ENDIF\nENDPROC\nENDMODULE\n")
    out = tmp_path / "w.csv"
    assert main([str(prog), "-o", str(out), "-q"]) == 0  # a run-time condition is an assumption, not missing data
    assert main([str(prog), "-o", str(out), "-q", "--strict"]) == 3
