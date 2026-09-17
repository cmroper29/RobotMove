import numpy as np
import pytest

from robotparse.geometry import ArcLeg, BlendPiece, DegenerateArc, LineLeg, SplineLeg
from robotparse.transforms import T_from_xyzabc


def P(x, y, z, a=0.0, b=90.0, c=0.0):
    return T_from_xyzabc(x, y, z, a, b, c)


def test_arc_through_via_point():
    arc = ArcLeg(P(700, 0, 0), P(550, 0, 0), [625, 75, 0])
    assert np.allclose(arc.center, [625, 0, 0]) and arc.radius == pytest.approx(75)
    assert np.degrees(arc.sweep) == pytest.approx(180)
    assert np.allclose(arc.pos(arc.length / 2)[0], [625, 75, 0])
    assert np.allclose(arc.end_T()[:3, 3], [550, 0, 0])


def test_arc_other_direction_and_excluded_reference():
    arc = ArcLeg(P(700, 0, 0), P(550, 0, 0), [625, -75, 0])
    assert np.allclose(arc.pos(arc.length / 2)[0], [625, -75, 0])
    # quarter arc from (600,75) to (550,125) on the circle centred (600,125) that avoids (600,175)
    arc = ArcLeg(P(600, 75, 0), P(550, 125, 0), [600, 175, 0], ref_excluded=True)
    assert np.degrees(arc.sweep) == pytest.approx(90)
    mid = arc.pos(arc.length / 2)[0]
    assert np.linalg.norm(mid - [600, 125, 0]) == pytest.approx(50)
    assert mid[0] < 600 and mid[1] < 125


def test_kuka_circular_angle():
    arc = ArcLeg(P(100, 0, 0), P(0, 100, 0), [70.7106781, 70.7106781, 0], circ_angle=270)
    assert np.degrees(arc.sweep) == pytest.approx(270)
    assert np.allclose(arc.end_T()[:3, 3], [0, -100, 0], atol=1e-6)


def test_collinear_arc_raises():
    with pytest.raises(DegenerateArc):
        ArcLeg(P(0, 0, 0), P(100, 0, 0), [50, 0, 0])


def test_line_orientation_interpolation():
    leg = LineLeg(P(0, 0, 0, a=0), P(100, 0, 0, a=90))
    R = leg.rot(50).as_matrix()[0]
    assert np.allclose(R, P(0, 0, 0, a=45)[:3, :3])
    only_rot = LineLeg(P(0, 0, 0, a=0), P(0, 0, 0, a=90))
    assert not only_rot.moves_position and only_rot.plen == pytest.approx(90)


def test_blend_is_tangent_continuous_and_stays_near_corner():
    A = LineLeg(P(0, 0, 0), P(100, 0, 0))
    B = LineLeg(P(100, 0, 0), P(100, 100, 0))
    d = 20.0
    blend = BlendPiece(A, A.plen - d, B, d, 0)
    s = np.linspace(0, blend.length, 400)
    pts = blend.pos(s)
    assert np.allclose(pts[0], [80, 0, 0]) and np.allclose(pts[-1], [100, 20, 0])
    steps = np.diff(pts, axis=0)
    assert np.allclose(np.linalg.norm(steps, axis=1) / np.diff(s), 1.0, atol=2e-3)  # arc-length parameter
    t0 = steps[0] / np.linalg.norm(steps[0])
    t1 = steps[-1] / np.linalg.norm(steps[-1])
    assert np.allclose(t0, [1, 0, 0], atol=1e-2) and np.allclose(t1, [0, 1, 0], atol=1e-2)
    corner_gap = np.min(np.linalg.norm(pts - [100, 0, 0], axis=1))
    assert 0 < corner_gap < d


def test_spline_passes_through_points():
    pts = [P(100, 100, 0), P(200, 0, 50), P(300, 100, 0)]
    leg = SplineLeg(P(0, 0, 0), pts)
    dense = leg.pos(np.linspace(0, leg.plen, 4000))
    for T in pts:
        assert np.min(np.linalg.norm(dense - T[:3, 3], axis=1)) < 0.5
    assert np.allclose(leg.end_T()[:3, 3], [300, 100, 0], atol=1e-6)
