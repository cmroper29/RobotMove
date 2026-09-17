import numpy as np

from robotparse.transforms import (T_from_pos_quat, T_from_xyzabc, T_from_xyzwpr, frame_from_spec, quat_wxyz_from_T,
                                   xyzabc_from_T, xyzwpr_from_T)


def test_kuka_b90_points_tool_x_down():
    T = T_from_xyzabc(1, 2, 3, 0, 90, 0)
    assert np.allclose(T[:3, 0], [0, 0, -1])
    assert np.allclose(T[:3, 3], [1, 2, 3])


def test_kuka_abc_is_intrinsic_zyx():
    T = T_from_xyzabc(0, 0, 0, 90, 0, 0)  # A rotates about Z
    assert np.allclose(T[:3, 0], [0, 1, 0])
    T = T_from_xyzabc(0, 0, 0, 90, 0, 90)  # then C about the new X (= world Y)
    assert np.allclose(T[:3, 0], [0, 1, 0])
    assert np.allclose(T[:3, 2], [1, 0, 0])


def test_fanuc_w180_points_tool_z_down():
    assert np.allclose(T_from_xyzwpr(0, 0, 0, 180, 0, 0)[:3, 2], [0, 0, -1])


def test_fanuc_wpr_equals_kuka_abc_reversed():
    a, b, c = 30, -20, 50
    assert np.allclose(T_from_xyzwpr(0, 0, 0, c, b, a), T_from_xyzabc(0, 0, 0, a, b, c))


def test_abb_quaternion_scalar_first():
    T = T_from_pos_quat([0, 0, 0], [0, 0, 1, 0])  # 180 deg about Y
    assert np.allclose(T[:3, 2], [0, 0, -1])
    assert np.allclose(np.abs(quat_wxyz_from_T(T)), [0, 0, 1, 0])


def test_roundtrips():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = np.concatenate([rng.uniform(-500, 500, 3), rng.uniform(-80, 80, 3)])
        assert np.allclose(xyzabc_from_T(T_from_xyzabc(*v)), v)
        assert np.allclose(xyzwpr_from_T(T_from_xyzwpr(*v)), v)


def test_frame_spec_forms():
    assert np.allclose(frame_from_spec({"xyz": [1, 2, 3], "abc": [0, 90, 0]}), T_from_xyzabc(1, 2, 3, 0, 90, 0))
    assert np.allclose(frame_from_spec({"xyz": [1, 2, 3], "quat": [0, 0, 1, 0]}), T_from_pos_quat([1, 2, 3], [0, 0, 1, 0]))
    assert np.allclose(frame_from_spec(None), np.eye(4))
