"""Animation timeline: every sampled frame is physically legal and inside the joint ranges."""
import numpy as np

from ghostsim.mechanism import H_MAX, WRIST_MIN_DEG
from tools.mech_anim import solids_at, timeline


def test_every_frame_is_legal():
    for t in np.arange(0.0, timeline(), 0.1):
        solids, st, _ = solids_at(t)
        assert -1e-4 < min(s.min_z() for s in solids), f"t={t:.2f} something below the floor"
        assert 0.0 <= st.h <= H_MAX + 1e-9
        assert WRIST_MIN_DEG - 1e-9 <= st.theta <= 90.0 + 1e-9
        assert -10.0 - 1e-9 <= st.psi <= 75.0 + 1e-9


def test_coral_released_at_l4_height_and_algae_taken_at_wrist_limit():
    from ghostsim.mechanism import EE_LEN, ee_wrist
    _, st, seg = solids_at(6.9)
    assert seg["name"].startswith("score coral")
    P, _ = ee_wrist(st.h)
    tip_z = P[2] + EE_LEN * np.sin(np.radians(st.theta))
    assert abs(tip_z - 72 * 0.0254) < 0.01                  # L4 branch top, within 1 cm
    _, st, seg = solids_at(10.8)
    assert seg["name"].startswith("algae into processor")
    assert abs(st.theta - WRIST_MIN_DEG) < 1e-6 and st.h == 0.0
