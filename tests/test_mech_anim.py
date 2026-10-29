"""Animation (tools/mech_anim.py, the live view's JointTracker on a script): every frame is legal and the coral leaves
the effector at L4 height."""
import numpy as np

from ghostsim.mechanism import H_MAX, coral_held
from ghostsim.mechanism_state import CORAL_AT
from tools.mech_anim import frames


def test_every_frame_is_legal_and_coral_released_at_l4():
    released, prev_coral = None, None
    for t, name, solids, tr in frames():
        assert min(s.min_z() for s in solids) > -1e-6, f"t={t:.2f} {name}: below the floor"
        h, w, c = tr.q[0]
        assert 0.0 <= h <= H_MAX + 1e-9
        has = any(s.name == "coral" and abs(s.center[2] - coral_held(h, w)[2]) < 1e-9 for s in solids)
        if prev_coral and not has and "released on L4" in name and released is None:
            released = coral_held(h, w)[2]
        prev_coral = has
    assert released is not None and abs(released - CORAL_AT["score_L4"][0]) < 0.02
    assert np.isfinite(released)
