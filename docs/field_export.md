# Field geometry: collision vs visual

## Collision geometry (physics, occlusion, rules) — source of truth
- `game/2025/game.json` (inches, values from the manual and FE-2025) + `apriltags-welded.json` (WPILib).
- `rules.py` derives everything else: field boundary = 4 walls + coral-station chamfers through tags 1/2/12/13;
  reef hexagon centre/apothem from reef tags; zones, cages, coral marks, starting line from game.json.
- AndyMark fields: `SimCfg(tags="andymark")` swaps the tag file (positions differ slightly, Team Update 12).

## Visual geometry (renderer)
Current: a **procedural proxy scene** generated from the same definition (`Game.scene()` → Godot primitives:
carpet, walls, coral stations, processors, reef prisms + pipes, barge, cages, tape lines, tag panels with
tag36h11 bits). Because Python generates it, it is consistent with physics by construction
(checked by `smoke/smoke_consistency.py`).

Optional higher-fidelity model from the official CAD (NOT implemented yet — needs a GLB loader in
`main.gd` and `tools/check_field.py`):
1. Download the official 2025 field CAD from the FIRST "REEFSCAPE Playing FIELD" page.
2. Import into Blender (STEP importer add-on, or FreeCAD → glTF). I have not verified which importer
   handles this assembly best, or whether Onshape's glTF export is usable — test both.
3. Simplify: delete fasteners and small hardware, decimate to < 300 k triangles, assign the material
   classes used by the renderer (carpet, guardrail, alliance_wall, station, processor, reef_base,
   reef_pipe, barge, truss, tape_*). Do not include tags (the renderer places them from the tag file).
4. Put the origin at the WPILib field origin (blue alliance-wall diamond plate, scoring-table border),
   +X toward red, +Z up in Blender (the glTF exporter converts to Y-up).
5. Name obstacle meshes `col_<poly_id>` (`reef_blue`, `reef_red`, `barge_support`, `cage_0..5`).
6. Export `game/2025/visual/field.glb` (use git LFS).
7. Verify: `tools/check_field.py` (to be written) compares `col_*` bounds with `Game.polys` (≤ 2 cm),
   then run `smoke/smoke_consistency.py`.
