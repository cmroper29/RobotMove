"""File reading, input validation, config errors and the readiness report."""
import codecs
import re
import shutil
import zipfile

import numpy as np
import pytest

from robotparse import load_config, parse_program
from robotparse.cli import main
from robotparse.model import RobotParseError
from robotparse.parsers import detect_vendor
from robotparse.textio import read_program_text

from .conftest import EXAMPLES, URDF

PROGRAMS = {
    "kuka": ["kuka/weld_demo.src", "kuka/weld_demo.dat"],
    "abb": ["abb/demo.mod"],
    "fanuc": ["fanuc/demo.ls"],
}
ENCODINGS = {
    "bom": lambda b: codecs.BOM_UTF8 + b,
    "utf16-bom": lambda b: b.decode().encode("utf-16"),
    "utf16-le-no-bom": lambda b: b.decode().encode("utf-16-le"),
    "crlf": lambda b: b.replace(b"\n", b"\r\n"),
}


def _copy(tmp_path, vendor, transform=lambda b: b):
    files = []
    for rel in PROGRAMS[vendor]:
        dst = tmp_path / rel.split("/")[1]
        dst.write_bytes(transform((EXAMPLES / rel).read_bytes()))
        files.append(dst)
    return files[0]


def _targets(program):
    return [np.round((m.base @ m.target.T)[:3, 3], 6).tolist() if hasattr(m.target, "T") else None
            for m in program.motions]


@pytest.mark.parametrize("vendor", PROGRAMS)
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_encodings_parse_identically(tmp_path, vendor, encoding):
    (tmp_path / "plain").mkdir()
    (tmp_path / "variant").mkdir()
    plain = parse_program([_copy(tmp_path / "plain", vendor)], load_config())
    variant = parse_program([_copy(tmp_path / "variant", vendor, ENCODINGS[encoding])], load_config())
    assert _targets(variant) == _targets(plain)
    assert len(variant.motions) == len(plain.motions) > 0


def test_windows_1252_comments(tmp_path):
    p = tmp_path / "umlaut.src"
    p.write_bytes("DEF umlaut()\n; Schweißnaht Überprüfung\nLIN {X 1, Y 2, Z 3, A 0, B 0, C 0}\nEND\n".encode("cp1252"))
    assert len(parse_program([p], load_config()).motions) == 1


def test_binary_archive_folder_and_tp_messages(tmp_path):
    tp = tmp_path / "PROG.tp"
    tp.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(RobotParseError, match=r"\.ls"):
        detect_vendor(tp)
    blob = tmp_path / "prog.mod"
    blob.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(RobotParseError, match="not a text program"):
        read_program_text(blob)
    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.src", "DEF a()\nEND\n")
    with pytest.raises(RobotParseError, match="Extract it"):
        read_program_text(archive)
    with pytest.raises(RobotParseError, match="folder"):
        parse_program([tmp_path], load_config(), vendor="kuka")
    with pytest.raises(RobotParseError, match="not found"):
        parse_program([tmp_path / "missing.src"], load_config())


def test_content_sniffing_and_not_a_program(tmp_path):
    txt = tmp_path / "program.txt"
    shutil.copy(EXAMPLES / "fanuc" / "demo.ls", txt)
    assert detect_vendor(txt) == "fanuc"
    junk = tmp_path / "notes.txt"
    junk.write_text("shopping list\n")
    with pytest.raises(RobotParseError, match="cannot tell which robot language"):
        detect_vendor(junk)
    empty = tmp_path / "empty.ls"
    empty.write_text("")
    with pytest.raises(RobotParseError, match="no /MN section"):
        parse_program([empty], load_config())


def test_abb_pgf_expands_to_modules(tmp_path):
    shutil.copy(EXAMPLES / "abb" / "demo.mod", tmp_path / "MainModule.mod")
    pgf = tmp_path / "Program.pgf"
    pgf.write_text('<?xml version="1.0"?>\n<Program>\n  <Module>MainModule.mod</Module>\n</Program>\n')
    assert len(parse_program([pgf], load_config()).motions) == 13


# ----------------------------------------------------------------------------- config validation
def test_config_errors_are_readable(tmp_path):
    cases = {
        "motion:\n  cart_acel: 3000\n": "Did you mean 'motion.cart_accel'",
        "tool:\n  1: {xyz: [0, 0, 1]}\n": "Did you mean 'tools'",
        "tools:\n  1: {xyz: [0, 150], abc: [0, 0, 0]}\n": r"tools\.1\.xyz: expected a list of 3 numbers",
        "tools:\n  1: {xyz: [0, 0, 150], abc: [0, 0, 0]\n": r"invalid YAML \(line",
        "tools:\n  1: {xyz: [TODO, 0, 0], abc: [0, 0, 0]}\nrobot:\n  urdf: TODO\n": r"TODO placeholders.*tools\.1, robot\.urdf",
        "units: inches\n": "units must be",
    }
    for i, (text, pattern) in enumerate(cases.items()):
        p = tmp_path / f"c{i}.yaml"
        p.write_text(text)
        with pytest.raises(RobotParseError, match=pattern):
            load_config(p)


def test_multiple_configs_merge_in_order(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("units: m\ntools:\n  1: {xyz: [0, 0, 100], abc: [0, 0, 0]}\n")
    b.write_text("tools:\n  2: {xyz: [0, 0, 200], abc: [0, 0, 0]}\nmotion:\n  cart_accel: 4000\n")
    cfg = load_config([a, b])
    assert cfg["units"] == "m" and set(cfg["tools"]) == {1, 2} and cfg["motion"]["cart_accel"] == 4000
    assert "motion.cart_accel" in cfg["_user_set"] and "world_from_robot" not in cfg["_user_set"]


# ----------------------------------------------------------------------------- CLI behaviour and the report
def test_errors_have_no_traceback(tmp_path, capsys):
    junk = tmp_path / "junk.src"
    junk.write_text("hello\n")
    assert main([str(junk), "-o", str(tmp_path / "o.csv")]) == 1
    err = capsys.readouterr().err
    assert "error: no DEF" in err and "Traceback" not in err


def test_program_without_motion_fails_and_writes_nothing(tmp_path, capsys):
    prog = tmp_path / "empty.mod"
    prog.write_text('MODULE E\nPROC main()\nTPWrite "hello";\nENDPROC\nENDMODULE\n')
    out = tmp_path / "o.csv"
    assert main([str(prog), "-o", str(out)]) == 1
    assert not out.exists()
    assert "FAILED" in capsys.readouterr().out


def test_report_template_round_trip(tmp_path, capsys):
    prog = _copy(tmp_path, "fanuc")
    out = tmp_path / "demo.csv"
    assert main([str(prog), "--dt", "0.05", "-o", str(out)]) == 4
    report = capsys.readouterr().out
    assert "Motions    12 of 14 used, 2 skipped" in report
    for item in ("UTOOL 1", "UFRAME 1", "PR[1]", "PR[2]", "Robot model (URDF)"):
        assert item in report
    template = tmp_path / "demo_missing.yaml"
    text = template.read_text()
    assert text.count("TODO") >= 10

    assert main([str(prog), "--dt", "0.05", "-o", str(out), "-c", str(template)]) == 1  # TODO left in
    assert "TODO placeholders" in capsys.readouterr().err

    fills = {r"tools:\n(  #.*\n)  1: .*": "tools:\n  1: {xyz: [0, 0, 150], wpr: [0, 0, 0]}",
             r"bases:\n(  #.*\n)  1: .*": "bases:\n  1: {xyz: [400, 0, 0], wpr: [0, 0, 0]}",
             r"    1: \{xyz: \[TODO.*": "    1: {xyz: [0, 0, 100], wpr: [0, 0, 0]}",
             r"    2: \{xyz: \[TODO.*": "    2: {xyz: [0, -75, 0], wpr: [0, 0, 0]}",
             r"urdf: TODO.*": f"urdf: {URDF}", r"tip_link: TODO.*": "tip_link: tool0"}
    for pattern, value in fills.items():
        text = re.sub(pattern, value, text)
    filled = tmp_path / "filled.yaml"
    filled.write_text(text)
    template.unlink()
    assert main([str(prog), "--dt", "0.05", "-o", str(out), "-c", str(filled)]) == 0
    report = capsys.readouterr().out
    assert "Motions    14 of 14 used" in report and "COMPLETE" in report
    assert not template.exists()


def test_kuka_missing_dat_is_reported(tmp_path, capsys):
    shutil.copy(EXAMPLES / "kuka" / "weld_demo.src", tmp_path / "weld_demo.src")
    assert main([str(tmp_path / "weld_demo.src"), "-o", str(tmp_path / "w.csv"), "--no-template"]) == 4
    report = capsys.readouterr().out
    assert "File weld_demo.dat was not provided" in report
    assert "uses data that is not defined in the loaded files" in report
    assert not (tmp_path / "w_missing.yaml").exists()


def test_template_is_never_overwritten_and_quiet_prints_verdict(tmp_path, capsys):
    prog = _copy(tmp_path, "fanuc")
    out = tmp_path / "demo.csv"
    template = tmp_path / "demo_missing.yaml"
    template.write_text("# my edits\n")
    assert main([str(prog), "-o", str(out)]) == 4
    assert template.read_text() == "# my edits\n"
    assert "not overwritten" in capsys.readouterr().out
    assert main([str(prog), "-o", str(out), "-q"]) == 4
    assert capsys.readouterr().out.strip().startswith("INCOMPLETE")
