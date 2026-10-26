import numpy as np

from ghostsim import geometry as geo

H = np.array([0.5, 0.5])
c0, s0 = np.array(1.0), np.array(0.0)


def test_obb_sat_basic_and_symmetry():
    d, n, k = geo.obb_sat(np.zeros(2), H, c0, s0, np.array([0.8, 0.1]), H, c0, s0)
    assert np.isclose(d, 0.2) and np.allclose(n, [1, 0])
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1, 1, (200, 2)), rng.uniform(-1, 1, (200, 2))
    ya, yb = rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200)
    d1, n1, _ = geo.obb_sat(a, H, np.cos(ya), np.sin(ya), b, H, np.cos(yb), np.sin(yb))
    d2, n2, _ = geo.obb_sat(b, H, np.cos(yb), np.sin(yb), a, H, np.cos(ya), np.sin(ya))
    assert np.allclose(d1, d2) and np.allclose(n1[d1 > 1e-6], -n2[d1 > 1e-6])


def test_box_poly_matches_obb_for_rectangles():
    rng = np.random.default_rng(1)
    poly = geo.Polygons([geo.rect(0.3, -0.2, 1.0, 0.6, 0.4)])
    c = rng.uniform(-1, 1, (300, 2))
    y = rng.uniform(-3, 3, 300)
    d_poly, n_poly, _ = geo.box_poly_sat(c, H, np.cos(y), np.sin(y), poly, margin=10)
    d_obb, n_obb, _ = geo.obb_sat(c, H, np.cos(y), np.sin(y), np.array([0.3, -0.2]), np.array([0.5, 0.3]),
                                  np.cos(0.4), np.sin(0.4))
    hit = d_obb > 1e-6
    assert np.allclose(d_poly[:, 0], d_obb, atol=1e-9)
    assert np.allclose(n_poly[hit, 0], n_obb[hit], atol=1e-9)


def test_hexagon_and_broadphase():
    hexa = geo.regular_polygon([0, 0], 1.0, 6, 0.0)
    assert np.allclose(np.linalg.norm(hexa, axis=1), 1 / np.cos(np.pi / 6))
    poly = geo.Polygons([hexa])
    d, n, p = geo.box_poly_sat(np.array([[1.4, 0.0]]), H, np.ones(1), np.zeros(1), poly)
    assert np.isclose(d[0, 0], 0.1) and np.allclose(n[0, 0], [-1, 0])
    d, _, _ = geo.box_poly_sat(np.array([[5.0, 0.0]]), H, np.ones(1), np.zeros(1), poly)
    assert d[0, 0] == -np.inf


def test_contact_point_is_deepest_corner():
    poly = geo.Polygons([geo.rect(0, 0, 2, 2)])
    yaw = np.pi / 4
    c = np.array([[1.0 + 0.5 * np.sqrt(2) - 0.1, 0.0]])
    d, n, p = geo.box_poly_sat(c, H, np.array([np.cos(yaw)]), np.array([np.sin(yaw)]), poly)
    assert np.isclose(d[0, 0], 0.1, atol=1e-9) and np.allclose(p[0, 0], [0.9, 0.0], atol=1e-9)


def test_halfplane_circle_segment():
    n, off = np.array([[1.0, 0.0]]), np.array([0.0])
    d, pt = geo.halfplane_box(np.array([0.3, 0.0]), H, c0, s0, n, off)
    assert np.isclose(d[0], 0.2) and np.isclose(pt[0, 0], -0.2)
    dd, nn = geo.circle_box(np.array([[0.7, 0.0]]), 0.3, np.zeros((1, 2)), H[None], np.ones(1), np.zeros(1))
    assert np.isclose(dd[0, 0], 0.1) and np.allclose(nn[0, 0], [1, 0])
    assert geo.segment_box_hit(np.array([-2.0, 0.1]), np.array([2.0, 0.1]), np.zeros(2), H, c0, s0)
    assert not geo.segment_box_hit(np.array([-2.0, 0.9]), np.array([2.0, 0.9]), np.zeros(2), H, c0, s0)
