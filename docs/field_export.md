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
`ghostrender` uses the optimised CAD model when `game/2025/visual/field.glb` exists, otherwise the
procedural proxy scene generated from the same definition (`SimCfg`-independent, `visual="auto"|"cad"|"proxy"`).

### Regenerating the CAD visual (source: official Onshape GLB, 947 k unique / 5.3 M drawn triangles)
```
uv run python tools/prepare_field.py            # finds REEFSCAPE FIELD*.glb in the repo root, ~10 s
uv run python tools/check_field.py              # cross-checks CAD vs tags / cages / barge support / reef
```
`prepare_field.py` removes AprilTags (FE-00083), the six CAGE assemblies, leftover CORAL/ALGAE, the scoring table
and parts < 4 cm; merges the ~54 000 remaining instances into one mesh per material (carpet is separate so
domain randomisation can recolour it), decimates to ~300 k triangles, and writes Y-up glTF.
The source (51 MB) and the output (~30 MB) are not committed (add git LFS if the team wants them in the repo).

Source frame (verified against the tag file to 5 mm): metres, Z up, origin at the field centre, +X toward red.
Quality notes: thin reef pipes get coarse at this budget (invisible at 160x120); the polycarbonate guardrail is
modelled opaque in the source, so cameras cannot see through it (they can on the real field).
Rendering cost: 24 cameras at 160x120 -> ~500 images/s with CAD vs ~720 with the proxy scene.
