# Field geometry: collision vs visual

## Collision geometry (physics, occlusion, rules) — source of truth
- `game/2025/game.json` (inches, from the manual and FE-2025) + `apriltags-welded.json` (WPILib).
- `rules.py` derives the rest: boundary = 4 walls + coral-station chamfers through tags 1/2/12/13;
  reef hexagon from the reef tags; zones, cages, coral marks, starting line from game.json.
- AndyMark fields: `SimCfg(tags="andymark")` swaps the tag file (Team Update 12).
- Values measured from the official CAD (see `tools/check_field.py`): barge centre support 12 x 12 in,
  guardrail inner faces 2.7 cm inside the 317 in width, cage layout (middle cage shallow, outer two deep),
  staged reef ALGAE rest pose (`field.reef.algae_rest`: cradled by the L2 / L3 branch pair, 5.7 in behind the tag
  plane, centre 36.3 / 52.1 in high; checked: < 1 cm from the CAD cradle, no penetration) and the trough slots for
  L1 CORAL (`field.reef.trough_slots`: 3 end to end per layer, 2 layers; a 3rd layer would hit the L2 branches).

## Visual geometry (renderer)
`ghostrender` uses the optimised CAD model when `game/2025/visual/field.glb` exists, otherwise the procedural proxy
scene generated from the same definition (`visual="auto"|"cad"|"proxy"`, or a path to a .glb). `visual="hi"` loads
the detailed `field_hi.glb` when present; the live view (`ghosttrain/broadcast.py`) uses it, robot-camera training
uses `field.glb`.

### Regenerating the CAD visual (source: official Onshape GLB, 947 k unique / 2.26 M drawn triangles >= 4 cm)
```
uv run python tools/prepare_field.py "REEFSCAPE FIELD (FE-2025) (1).glb"   # ~25 s, writes field.glb and field_hi.glb
uv run python tools/check_field.py                                          # CAD vs tags / cages / barge / reef / algae
```
`prepare_field.py` removes AprilTags (FE-00083), the six CAGE assemblies, leftover CORAL/ALGAE, the scoring table
and parts < 4 cm; decimates every unique part with one ratio, raised to a floor for thin long parts (pipes, rods:
64 / 24 triangles in field_hi / field, other parts 12 / 4), chosen so that the drawn total meets the budget
(`--budget` 300 k for field.glb, `--budget-hi` 1.5 M for field_hi.glb); gives every part crease-angle normals
(35 deg: cylinders smooth, boxes sharp); merges the instances into one mesh per material (carpet separate so
domain randomisation can recolour it) and writes Y-up glTF with double-sided materials, as in the source.
The source (51 MB) and the outputs (15 / 54 MB) are not committed (add git LFS if the team wants them in the repo).

Source frame (verified against the tag file to 5 mm): metres, Z up, origin at the field centre, +X toward red.

Fixed on 2026-10-29 (the field looked broken): the old export flipped every triangle after the axis change, which is
a rotation (det +1) and keeps the winding, and recomputed normals from the flipped triangles; with single-sided
materials Godot culled the outside of every part and drew its inside (coral station, processor and reef faces looked
hollow or missing). The single 6 % decimation ratio also collapsed the reef pipes to slivers. Regression test:
`tests/test_prepare_field.py`. A single-sided export renders faster but leaves holes where decimation opened thin
parts, so the materials stay double-sided.
Quality notes: the polycarbonate guardrail is modelled opaque in the source (no transparency in the CAD materials),
so cameras cannot see through it (they can on the real field).
Rendering cost (GTX 1070, 6 envs x 4 cameras at 160x120, simulation stepping included): old field.glb 301 images/s,
new field.glb 294, field_hi.glb 207. Live view 1280x720 with field_hi.glb: 24 fps (28.5 with the old, broken field).
