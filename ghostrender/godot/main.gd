## GhostADS renderer. Render-only: builds the scene from the Python scene description, places
## objects exactly where Python says, applies domain-randomisation values sampled by Python,
## renders the requested cameras and returns RGB images. No physics, no game logic.
## Field frame (WPILib, metres) -> Godot: (x, y, z) -> (x, z, -y).
extends Node

const MAGIC := "GHST"
const BUMPER_T := 0.08
const SLOTS := 64                 # solids per robot (ghostsim/mechanism.py SLOTS)
const SLOT_W := 20                # floats per solid: visible, kind, centre(3), half|radius(3), basis cols(9), colour(3)
const KIND_BODY := 3              # kind 3 + i: robot.glb body i (ghostsim/mechanism.py BODIES), basis = full linear map
const FONTS := ["DejaVu Sans", "Liberation Sans", "Noto Sans", "FreeSans", "Ubuntu", "DejaVu Serif",
		"Liberation Serif", "Noto Sans Mono", "DejaVu Sans Mono", "Cantarell"]

var server := TCPServer.new()
var peer: StreamPeerTCP
var buf := PackedByteArray()
var busy := false
var W := 160
var H := 120
var envs: Array = []
var cams: Array = []
var rigs: Array = []
var atlas: SubViewport
var atlas_cols := 1
var pieces := {}
var tag_bits := {}
var post_shader: Shader
var visual_template: Node3D
var robot_template: Node3D        # robot.glb (tools/prepare_robot.py): one child node per body
var robot_bodies: Array = []


func _ready() -> void:
	var port := 47800
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--port="):
			port = int(a.substr(7))
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	post_shader = load("res://post.gdshader")
	var err := server.listen(port, "127.0.0.1")
	print("GHOST_RENDER_LISTEN ", port, " ", err)


func _process(_dt: float) -> void:
	if peer == null:
		if server.is_connection_available():
			peer = server.take_connection()
			peer.set_no_delay(true)
		return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		get_tree().quit()
		return
	var n := peer.get_available_bytes()
	if n > 0:
		buf.append_array(peer.get_data(n)[1])
	if busy or buf.size() < 12:
		return
	var hl := buf.decode_u32(4)
	var bl := buf.decode_u32(8)
	if buf.size() < 12 + hl + bl:
		return
	var header: Dictionary = JSON.parse_string(buf.slice(12, 12 + hl).get_string_from_utf8())
	var blob := buf.slice(12 + hl, 12 + hl + bl)
	buf = buf.slice(12 + hl + bl)
	match header["type"]:
		"configure":
			configure(header)
			reply({"type": "ok"})
		"appearance":
			appearance(header)
			reply({"type": "ok"})
		"render":
			render(header, blob)
		"quit":
			reply({"type": "ok"})
			get_tree().quit()


func reply(header: Dictionary, blob := PackedByteArray()) -> void:
	var h := JSON.stringify(header).to_utf8_buffer()
	var out := MAGIC.to_ascii_buffer()
	var lens := PackedByteArray()
	lens.resize(8)
	lens.encode_u32(0, h.size())
	lens.encode_u32(4, blob.size())
	out.append_array(lens)
	out.append_array(h)
	out.append_array(blob)
	peer.put_data(out)


# ---------------------------------------------------------------- coordinates
func gv(v) -> Vector3:
	return Vector3(v[0], v[2], -v[1])


func gbasis_from_field(m: Array) -> Basis:
	## m: 3x3 row-major field rotation (columns = local x,y,z axes in field coords).
	## Godot local axes: X = local x, Y = local z, Z = -local y.
	var cx := gv([m[0][0], m[1][0], m[2][0]])
	var cy := gv([m[0][1], m[1][1], m[2][1]])
	var cz := gv([m[0][2], m[1][2], m[2][2]])
	return Basis(cx, cz, -cy)


func mat(rgb: Color, rough := 0.8, metal := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = rgb
	m.roughness = rough
	m.metallic = metal
	return m


func mesh_node(mesh: Mesh, m: Material, xf := Transform3D.IDENTITY) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = m
	mi.transform = xf
	return mi


# ---------------------------------------------------------------- configure
const MAT_COLORS := {
	"carpet": Color(0.22, 0.22, 0.23), "guardrail": Color(0.75, 0.78, 0.8), "alliance_wall": Color(0.55, 0.57, 0.6),
	"station": Color(0.7, 0.72, 0.75), "processor": Color(0.7, 0.72, 0.75), "reef_base": Color(0.33, 0.33, 0.35),
	"reef_pipe": Color(0.6, 0.15, 0.75), "barge": Color(0.45, 0.46, 0.48), "truss": Color(0.78, 0.79, 0.8),
	"tape_white": Color(0.9, 0.9, 0.9), "tape_black": Color(0.05, 0.05, 0.05), "tape_blue": Color(0.1, 0.25, 0.85),
	"tape_red": Color(0.85, 0.12, 0.1)}


func configure(h: Dictionary) -> void:
	W = int(h["width"])
	H = int(h["height"])
	pieces = h["pieces"]
	if visual_template:
		visual_template.queue_free()
		visual_template = null
	var glb: String = h.get("visual_glb", "")
	if glb != "":
		var doc := GLTFDocument.new()
		var gst := GLTFState.new()
		var gerr := doc.append_from_file(glb, gst)
		if gerr != OK:
			push_error("GLB load failed: %s (%d)" % [glb, gerr])
		else:
			visual_template = doc.generate_scene(gst)
			print("GHOST_RENDER_GLB_OK ", glb)
	if robot_template:
		robot_template.queue_free()
		robot_template = null
	robot_bodies = h.get("robot_bodies", [])
	var rglb: String = h.get("robot_glb", "")
	if rglb != "":
		var rdoc := GLTFDocument.new()
		var rst := GLTFState.new()
		if rdoc.append_from_file(rglb, rst) != OK:
			push_error("robot GLB load failed: %s" % rglb)
		else:
			robot_template = rdoc.generate_scene(rst)
			print("GHOST_RENDER_ROBOT_OK ", rglb)
	rigs = h["rigs"]
	tag_bits = h["tag_bits"]
	for e in envs:
		e["vp"].queue_free()
	for c in cams:
		c["vp"].queue_free()
	if atlas:
		atlas.queue_free()
	envs.clear()
	cams.clear()
	var tags_tex := {}
	for k in tag_bits.keys():
		var bits: Array = tag_bits[k]
		var data := PackedByteArray()
		for row in bits:
			for v in row:
				data.append(255 if int(v) == 1 else 0)
		var img := Image.create_from_data(8, 8, false, Image.FORMAT_L8, data)
		tags_tex[str(k)] = ImageTexture.create_from_image(img)
	for i in int(h["n_envs"]):
		envs.append(build_env(h["scene"], tags_tex))
	var k := int(h["max_cams"])
	atlas_cols = mini(k, 8)
	var rows := int(ceil(float(k) / atlas_cols))
	atlas = SubViewport.new()
	atlas.size = Vector2i(atlas_cols * W, rows * H)
	atlas.disable_3d = true
	atlas.transparent_bg = false
	atlas.render_target_update_mode = SubViewport.UPDATE_DISABLED
	add_child(atlas)
	for j in k:
		var vp := SubViewport.new()
		vp.size = Vector2i(W, H)
		vp.msaa_3d = [Viewport.MSAA_DISABLED, Viewport.MSAA_2X, Viewport.MSAA_4X, Viewport.MSAA_8X][clampi(int(h.get("msaa", 0)), 0, 3)]
		if bool(h.get("fxaa", false)):                # softer edges for the live view (thin pipes, nets, rails)
			vp.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		var cam := Camera3D.new()
		cam.projection = Camera3D.PROJECTION_FRUSTUM
		cam.keep_aspect = Camera3D.KEEP_HEIGHT
		cam.current = true
		vp.add_child(cam)
		add_child(vp)
		var rect := ColorRect.new()
		rect.position = Vector2((j % atlas_cols) * W, (j / atlas_cols) * H)
		rect.size = Vector2(W, H)
		var sm := ShaderMaterial.new()
		sm.shader = post_shader
		sm.set_shader_parameter("src", vp.get_texture())
		rect.material = sm
		atlas.add_child(rect)
		cams.append({"vp": vp, "cam": cam, "rect": rect, "mat": sm})


func build_env(scene: Dictionary, tags_tex: Dictionary) -> Dictionary:
	var vp := SubViewport.new()
	vp.own_world_3d = true
	vp.size = Vector2i(2, 2)
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	add_child(vp)
	var mats := {}
	for k in MAT_COLORS:
		mats[k] = mat(MAT_COLORS[k], 0.85, 0.3 if k in ["truss", "station", "processor", "guardrail"] else 0.0)
	var stat := Node3D.new()
	vp.add_child(stat)
	if visual_template:
		var vis := visual_template.duplicate()
		stat.add_child(vis)
		for c in vis.get_children():
			if c is MeshInstance3D:
				if c.name.begins_with("carpet"):
					c.material_override = mats["carpet"]
				else:
					(c as MeshInstance3D).cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	for p in scene["primitives"]:
		stat.add_child(build_primitive(p, mats, tags_tex))
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	vp.add_child(env)
	var sun := DirectionalLight3D.new()
	sun.shadow_enabled = true
	vp.add_child(sun)
	var dyn := Node3D.new()
	vp.add_child(dyn)
	var robots := []
	var proxies := []
	var mech_pool := []
	var bodies := []
	for r in 6:
		var n := Node3D.new()
		dyn.add_child(n)
		robots.append(n)
		var px := Node3D.new()                    # randomised box robot (appearance), drawn when no "mech" comes
		n.add_child(px)
		proxies.append(px)
		mech_pool.append(make_mech_pool(n))
		bodies.append(make_bodies(n))
	var coral := multimesh(CylinderMesh.new(), int(pieces["mc"]), mat(Color(0.95, 0.95, 0.93), 0.6), false)
	var cm: CylinderMesh = coral.multimesh.mesh
	cm.top_radius = pieces["coral_od"] / 2.0
	cm.bottom_radius = pieces["coral_od"] / 2.0
	cm.height = pieces["coral_len"]
	cm.radial_segments = 12
	dyn.add_child(coral)
	var algae := multimesh(SphereMesh.new(), int(pieces["ma"]), mat(Color(0.2, 0.75, 0.55), 0.7), false)
	var sm: SphereMesh = algae.multimesh.mesh
	sm.radius = pieces["algae_d"] / 2.0
	sm.height = pieces["algae_d"]
	dyn.add_child(algae)
	var cages := []
	for c in 6:
		var bm := BoxMesh.new()
		bm.size = Vector3(pieces["cage_size"], pieces["cage_h"], pieces["cage_size"])
		var mi := mesh_node(bm, mat(Color(0.1, 0.2, 0.8) if c < 3 else Color(0.8, 0.1, 0.1), 0.5, 0.4))
		dyn.add_child(mi)
		cages.append(mi)
	var unshaded := StandardMaterial3D.new()
	unshaded.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	unshaded.vertex_color_use_as_albedo = true
	var markers := multimesh(SphereMesh.new(), 0, unshaded, true)
	vp.add_child(markers)
	var crowd := multimesh(BoxMesh.new(), 0, mat(Color.WHITE, 0.9), true)
	crowd.material_override.vertex_color_use_as_albedo = true
	vp.add_child(crowd)
	return {"vp": vp, "world": vp.find_world_3d(), "static": stat, "dyn": dyn, "robots": robots, "proxy": proxies,
			"mech": mech_pool, "bodies": bodies, "coral": coral,
			"algae": algae, "cages": cages, "markers": markers, "crowd": crowd, "env": env, "sun": sun, "mats": mats}


func multimesh(mesh: Mesh, count: int, m: Material, colors: bool) -> MultiMeshInstance3D:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = colors
	mm.mesh = mesh
	mm.instance_count = count
	var mi := MultiMeshInstance3D.new()
	mi.multimesh = mm
	mi.material_override = m
	mi.extra_cull_margin = 100.0
	return mi


func build_primitive(p: Dictionary, mats: Dictionary, tags_tex: Dictionary) -> Node3D:
	match p["kind"]:
		"box":
			var bm := BoxMesh.new()
			var s: Array = p["size"]
			bm.size = Vector3(s[0], s[2], s[1])
			return mesh_node(bm, mats[p["mat"]], Transform3D(Basis(Vector3.UP, p["rpy"][2]), gv(p["pos"])))
		"cylinder":
			var cm := CylinderMesh.new()
			cm.top_radius = p["radius"]
			cm.bottom_radius = p["radius"]
			cm.height = p["height"]
			cm.radial_segments = 10
			return mesh_node(cm, mats[p["mat"]], Transform3D(Basis.IDENTITY, gv(p["pos"])))
		"prism":
			return mesh_node(prism(p["verts"], p["z"][0], p["z"][1]), mats[p["mat"]])
		"tag":
			var q: Array = p["quat_wxyz"]
			var b := Basis(Quaternion(q[1], q[2], q[3], q[0]))   # field-frame rotation (x = normal)
			var m := [[b.x.x, b.y.x, b.z.x], [b.x.y, b.y.y, b.z.y], [b.x.z, b.y.z, b.z.z]]
			var cx := gv([m[0][0], m[1][0], m[2][0]])
			var cy := gv([m[0][1], m[1][1], m[2][1]])
			var cz := gv([m[0][2], m[1][2], m[2][2]])
			var gb := Basis(cy, cz, cx)                 # quad: X = tag y (right), Y = tag z (up), Z = normal
			var node := Node3D.new()
			node.transform = Transform3D(gb, gv(p["pos"]))
			var panel := QuadMesh.new()
			panel.size = Vector2(p["panel"], p["panel"])
			node.add_child(mesh_node(panel, mat(Color(0.95, 0.95, 0.95), 0.9), Transform3D(Basis.IDENTITY, Vector3(0, 0, 0.003))))
			var quad := QuadMesh.new()
			quad.size = Vector2(p["size"], p["size"])
			var tm := mat(Color.WHITE, 0.9)
			tm.albedo_texture = tags_tex[str(int(p["id"]))]
			tm.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
			node.add_child(mesh_node(quad, tm, Transform3D(Basis.IDENTITY, Vector3(0, 0, 0.005))))
			return node
	return Node3D.new()


func prism(verts: Array, z0: float, z1: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var pts := []
	for v in verts:
		if pts.is_empty() or Vector2(v[0], v[1]).distance_to(pts[-1]) > 1e-6:
			pts.append(Vector2(v[0], v[1]))
	var c := Vector2.ZERO
	for v in pts:
		c += v
	c /= pts.size()
	for i in pts.size():
		var a: Vector2 = pts[i]
		var b: Vector2 = pts[(i + 1) % pts.size()]
		var e := b - a
		var nrm := gv([e.y, -e.x, 0.0]).normalized()
		for v in [[a, z0], [b, z0], [b, z1], [a, z0], [b, z1], [a, z1]]:
			st.set_normal(nrm)
			st.add_vertex(gv([v[0].x, v[0].y, v[1]]))
		for v in [[c, z1], [a, z1], [b, z1]]:
			st.set_normal(Vector3.UP)
			st.add_vertex(gv([v[0].x, v[0].y, v[1]]))
	var m := st.commit()
	return m


# ---------------------------------------------------------------- appearance / DR
## Robot mechanism: a fixed pool of unit boxes, spheres and cylinders per robot, posed every frame from the "mech" array.
func make_mech_pool(parent: Node3D) -> Array:
	var pool := []
	for i in SLOTS:
		var box := MeshInstance3D.new()
		box.mesh = BoxMesh.new()
		box.material_override = mat(Color.WHITE, 0.5, 0.3)
		box.visible = false
		parent.add_child(box)
		var sph := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = 0.5
		sm.height = 1.0
		sm.radial_segments = 16
		sm.rings = 8
		sph.mesh = sm
		sph.material_override = mat(Color.WHITE, 0.5, 0.3)
		sph.visible = false
		parent.add_child(sph)
		var cyl := MeshInstance3D.new()
		var cy := CylinderMesh.new()
		cy.top_radius = 0.5
		cy.bottom_radius = 0.5
		cy.height = 1.0
		cy.radial_segments = 16
		cy.rings = 1
		cyl.mesh = cy
		cyl.material_override = mat(Color.WHITE, 0.5, 0.3)
		cyl.visible = false
		parent.add_child(cyl)
		pool.append([box, sph, cyl])
	return pool


## Robot bodies: a copy of robot.glb per robot, one hidden node per body (meshes in robot-local Godot axes), posed
## every frame by the "mech" rows of kind KIND_BODY + i.
func make_bodies(parent: Node3D) -> Array:
	var out := []
	if robot_template == null:
		return out
	var inst := robot_template.duplicate()
	parent.add_child(inst)
	for name in robot_bodies:
		var b: Node3D = inst.find_child(str(name), true, false)
		if b == null:
			b = Node3D.new()
			inst.add_child(b)
		b.visible = false
		out.append(b)
	return out


func appearance(h: Dictionary) -> void:
	var e: Dictionary = envs[int(h["env"])]
	var rs: Array = h.get("robots", [])
	for r in mini(6, rs.size()):
		build_robot(e["proxy"][r], rs[r])
	var dr: Dictionary = h["dr"]
	var env: Environment = e["env"].environment
	env.background_color = Color(dr["bg_rgb"][0], dr["bg_rgb"][1], dr["bg_rgb"][2])
	env.ambient_light_color = Color(dr["ambient_rgb"][0], dr["ambient_rgb"][1], dr["ambient_rgb"][2])
	env.ambient_light_energy = dr["ambient_energy"]
	var sun: DirectionalLight3D = e["sun"]
	sun.light_energy = dr["sun_energy"]
	sun.light_color = Color(dr["sun_rgb"][0], dr["sun_rgb"][1], dr["sun_rgb"][2])
	sun.rotation = Vector3(-deg_to_rad(dr["sun_elev_deg"]), deg_to_rad(dr["sun_az_deg"]), 0.0)
	e["mats"]["carpet"].albedo_color = Color(dr["carpet_rgb"][0], dr["carpet_rgb"][1], dr["carpet_rgb"][2])
	e["mats"]["carpet"].roughness = dr["carpet_rough"]
	build_crowd(e["crowd"], int(dr["crowd_seed"]), dr["crowd_density"], h["field"])


## Randomised box robot (body, bumpers with the team number, a few boxes on top) for renders without a mechanism
## array (robot cameras in training, replays). `node` is the robot's proxy container, not the robot node itself (that
## also holds the mechanism pool and the robot.glb bodies).
func build_robot(node: Node3D, a: Dictionary) -> void:
	for c in node.get_children():
		c.queue_free()
	var lx: float = a["size"][0]
	var ly: float = a["size"][1]
	var hh: float = a["size"][2]
	var z0: float = a["bumper_z"][0]
	var z1: float = a["bumper_z"][1]
	var bc := Color(a["bumper_rgb"][0], a["bumper_rgb"][1], a["bumper_rgb"][2])
	var body := BoxMesh.new()
	body.size = Vector3(lx - 2 * BUMPER_T, hh - z0, ly - 2 * BUMPER_T)
	node.add_child(mesh_node(body, mat(Color(a["body_rgb"][0], a["body_rgb"][1], a["body_rgb"][2]), 0.5, 0.5),
			Transform3D(Basis.IDENTITY, Vector3(0, z0 + (hh - z0) / 2, 0))))
	var bm := mat(bc, 0.95)
	var zc := (z0 + z1) / 2
	for s in [[Vector3(lx / 2 - BUMPER_T / 2, zc, 0), Vector3(BUMPER_T, z1 - z0, ly)],
			[Vector3(-lx / 2 + BUMPER_T / 2, zc, 0), Vector3(BUMPER_T, z1 - z0, ly)],
			[Vector3(0, zc, ly / 2 - BUMPER_T / 2), Vector3(lx, z1 - z0, BUMPER_T)],
			[Vector3(0, zc, -ly / 2 + BUMPER_T / 2), Vector3(lx, z1 - z0, BUMPER_T)]]:
		var b := BoxMesh.new()
		b.size = s[1]
		node.add_child(mesh_node(b, bm, Transform3D(Basis.IDENTITY, s[0])))
	var font := SystemFont.new()
	font.font_names = PackedStringArray([FONTS[int(a["font"]) % FONTS.size()]])
	font.font_weight = 700 if int(a["font"]) % 2 == 0 else 500
	for side in [[Vector3(lx / 2 + 0.002, zc, 0), PI / 2, ly], [Vector3(-lx / 2 - 0.002, zc, 0), -PI / 2, ly],
			[Vector3(0, zc, ly / 2 + 0.002), 0.0, lx], [Vector3(0, zc, -ly / 2 - 0.002), PI, lx]]:
		var lab := Label3D.new()
		lab.text = str(a["team"])
		lab.font = font
		lab.font_size = 64
		lab.pixel_size = (z1 - z0) * 0.75 / 64.0
		lab.modulate = Color.WHITE
		lab.outline_size = 0
		lab.shaded = true
		lab.transform = Transform3D(Basis(Vector3.UP, side[1]), side[0])
		node.add_child(lab)
	var rng := RandomNumberGenerator.new()
	rng.seed = int(a["seed"])
	for i in rng.randi_range(1, 4):
		var b := BoxMesh.new()
		b.size = Vector3(rng.randf_range(0.05, lx * 0.6), rng.randf_range(0.1, 1.0), rng.randf_range(0.05, ly * 0.6))
		var g := rng.randf_range(0.2, 0.85)
		node.add_child(mesh_node(b, mat(Color(g, g, g * rng.randf_range(0.9, 1.1)), 0.4, 0.6),
				Transform3D(Basis.IDENTITY, Vector3(rng.randf_range(-lx, lx) * 0.25, hh + b.size.y / 2 - 0.05,
				rng.randf_range(-ly, ly) * 0.25))))


func build_crowd(mi: MultiMeshInstance3D, seed: int, density: float, field: Array) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var n := int(400 * density)
	var mm := mi.multimesh
	mm.instance_count = 0
	(mm.mesh as BoxMesh).size = Vector3(1, 1, 1)
	mm.instance_count = n
	var L: float = field[0]
	var Wd: float = field[1]
	for i in n:
		var side := rng.randi_range(0, 3)
		var d := rng.randf_range(2.5, 12.0)
		var t := rng.randf()
		var p: Vector2
		match side:
			0: p = Vector2(-d, t * (Wd + 20) - 10)
			1: p = Vector2(L + d, t * (Wd + 20) - 10)
			2: p = Vector2(t * (L + 20) - 10, -d)
			_: p = Vector2(t * (L + 20) - 10, Wd + d)
		var hgt := rng.randf_range(0.6, 1.9)
		var z := (d - 2.5) * 0.4 + hgt / 2
		var xf := Transform3D(Basis.from_scale(Vector3(rng.randf_range(0.3, 0.7), hgt, rng.randf_range(0.3, 0.7))), gv([p.x, p.y, z]))
		mm.set_instance_transform(i, xf)
		mm.set_instance_color(i, Color(rng.randf(), rng.randf(), rng.randf()))


# ---------------------------------------------------------------- render
func arrays_from(h: Dictionary, blob: PackedByteArray) -> Dictionary:
	var out := {}
	var off := 0
	for item in h["arrays"]:
		var count := 1
		for d in item[1]:
			count *= int(d)
		out[item[0]] = blob.slice(off, off + count * 4).to_float32_array()
		off += count * 4
	return out


func render(h: Dictionary, blob: PackedByteArray) -> void:
	busy = true
	var t0 := Time.get_ticks_usec()
	var a := arrays_from(h, blob)
	var check: bool = h.get("check", false)
	var echo := []
	var used := {}
	for c in h["cams"]:
		used[int(c[0])] = true
	for ei in used.keys():
		place_env(ei, a, check, echo)
	var cams_req: Array = h["cams"]
	for j in cams.size():
		var cam_d: Dictionary = cams[j]
		if j >= cams_req.size():
			cam_d["rect"].visible = false
			continue
		var req: Array = cams_req[j]
		var e: Dictionary = envs[int(req[0])]
		var vp: SubViewport = cam_d["vp"]
		vp.world_3d = e["world"]
		var rig_j: Dictionary = rigs[int(req[2])]
		var ov: Array = h.get("rig_override", [])          # per-render camera (e.g. a broadcast camera tracking a robot)
		if j < ov.size() and ov[j] != null:
			rig_j = ov[j]
		setup_camera(cam_d["cam"], a["robots"], int(req[0]), int(req[1]), rig_j)
		var pp: Array = h["post"][j]
		var sm: ShaderMaterial = cam_d["mat"]
		sm.set_shader_parameter("raw", check)
		sm.set_shader_parameter("gain", pp[0])
		sm.set_shader_parameter("noise", pp[1])
		sm.set_shader_parameter("blur", Vector2(pp[2] / W, pp[3] / H))
		sm.set_shader_parameter("seed", pp[4])
		cam_d["rect"].visible = true
		vp.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	atlas.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	var img := atlas.get_texture().get_image()
	img.convert(Image.FORMAT_RGB8)
	var ms := (Time.get_ticks_usec() - t0) / 1000.0
	reply({"type": "frame", "frame": h["frame"], "render_ms": ms, "atlas": [atlas_cols, atlas.size.y / H],
			"size": [W, H], "n": cams_req.size(), "echo": echo}, img.get_data())
	busy = false


func place_env(ei: int, a: Dictionary, check: bool, echo: Array) -> void:
	var e: Dictionary = envs[ei]
	var rob: PackedFloat32Array = a["robots"]
	for r in 6:
		var o := (ei * 6 + r) * 4
		var node: Node3D = e["robots"][r]
		e["proxy"][r].visible = not a.has("mech")         # the mechanism draws the robot when it is sent
		node.transform = Transform3D(Basis(Vector3.UP, rob[o + 2]), gv([rob[o], rob[o + 1], rob[o + 3]]))
		if check:
			var g := node.global_transform
			echo.append([ei, r, g.origin.x, -g.origin.z, g.origin.y, atan2(-g.basis.x.z, g.basis.x.x)])
	if a.has("mech"):
		var mm_: PackedFloat32Array = a["mech"]
		for r in 6:
			var pool: Array = e["mech"][r]
			var bods: Array = e["bodies"][r]
			for b in bods:
				b.visible = false
			for i in SLOTS:
				var o := ((ei * 6 + r) * SLOTS + i) * SLOT_W
				var box: MeshInstance3D = pool[i][0]
				var sph: MeshInstance3D = pool[i][1]
				var cyl: MeshInstance3D = pool[i][2]
				box.visible = false
				sph.visible = false
				cyl.visible = false
				if mm_[o] < 0.5:
					continue
				if mm_[o + 1] > KIND_BODY - 0.5:          # robot.glb body: full linear map, no colour
					var bi := int(mm_[o + 1] - KIND_BODY + 0.5)
					if bi < bods.size():
						var bn: Node3D = bods[bi]
						bn.transform = Transform3D(Basis(Vector3(mm_[o + 8], mm_[o + 11], mm_[o + 14]),
								Vector3(mm_[o + 9], mm_[o + 12], mm_[o + 15]), Vector3(mm_[o + 10], mm_[o + 13], mm_[o + 16])),
								Vector3(mm_[o + 2], mm_[o + 3], mm_[o + 4]))
						bn.visible = true
					continue
				var col := Color(mm_[o + 17], mm_[o + 18], mm_[o + 19])
				var origin := Vector3(mm_[o + 2], mm_[o + 3], mm_[o + 4])
				if mm_[o + 1] > 0.5 and mm_[o + 1] < 1.5:    # sphere (algae)
					sph.transform = Transform3D(Basis.from_scale(Vector3.ONE * 2.0 * mm_[o + 5]), origin)
					sph.visible = true
					(sph.material_override as StandardMaterial3D).albedo_color = col
				else:                                  # box or cylinder (axis = local Y): basis columns scaled by full size
					var bx := Vector3(mm_[o + 8], mm_[o + 11], mm_[o + 14]) * (2.0 * mm_[o + 5])
					var by := Vector3(mm_[o + 9], mm_[o + 12], mm_[o + 15]) * (2.0 * mm_[o + 6])
					var bz := Vector3(mm_[o + 10], mm_[o + 13], mm_[o + 16]) * (2.0 * mm_[o + 7])
					var mi: MeshInstance3D = box if mm_[o + 1] < 0.5 else cyl
					mi.transform = Transform3D(Basis(bx, by, bz), origin)
					mi.visible = true
					(mi.material_override as StandardMaterial3D).albedo_color = col
	var cor: PackedFloat32Array = a["coral"]
	var mm: MultiMesh = e["coral"].multimesh
	var nc := mm.instance_count
	for i in nc:
		var o := (ei * nc + i) * 6
		if cor[o + 5] < 0.5:
			mm.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3.ZERO), Vector3.ZERO))
		else:
			var b := Basis(Vector3.UP, cor[o + 3]) * Basis(Vector3(0, 0, 1), -(PI / 2 - cor[o + 4]))
			mm.set_instance_transform(i, Transform3D(b, gv([cor[o], cor[o + 1], cor[o + 2]])))
	var alg: PackedFloat32Array = a["algae"]
	mm = e["algae"].multimesh
	var na := mm.instance_count
	for i in na:
		var o := (ei * na + i) * 4
		var s := 1.0 if alg[o + 3] > 0.5 else 0.0
		mm.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3(s, s, s)), gv([alg[o], alg[o + 1], alg[o + 2]])))
	var cg: PackedFloat32Array = a["cages"]
	for c in 6:
		var o := (ei * 6 + c) * 4
		e["cages"][c].position = gv([cg[o], cg[o + 1], cg[o + 2]])
	e["static"].visible = not check
	e["dyn"].visible = not check
	e["crowd"].visible = not check
	var env: Environment = e["env"].environment
	if check:
		env.background_color = Color.BLACK
	var mk: MultiMesh = e["markers"].multimesh
	if a.has("markers"):
		var m: PackedFloat32Array = a["markers"]
		var per := int(m.size() / (4 * envs.size()))
		mk.instance_count = per
		for i in per:
			var o := (ei * per + i) * 4
			var rr := m[o + 3]
			mk.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3(rr * 2, rr * 2, rr * 2)), gv([m[o], m[o + 1], m[o + 2]])))
			mk.set_instance_color(i, Color.from_hsv(float(i) / per, 1.0, 1.0))
	else:
		mk.instance_count = 0


func setup_camera(cam: Camera3D, rob: PackedFloat32Array, ei: int, r: int, rig: Dictionary) -> void:
	var o := (ei * 6 + r) * 4
	var x := rob[o]
	var y := rob[o + 1]
	var yaw := rob[o + 2]
	var z := rob[o + 3]
	var c := cos(yaw)
	var s := sin(yaw)
	var t: Array = rig["t"]
	var m: Array = rig["R"]                      # robot <- camera (WPILib camera: x fwd, y left, z up)
	var pos: Array
	var f := []                                  # field <- camera
	if rig.get("world", false):                  # fixed spectator camera: t / R are given in the field frame
		pos = [t[0], t[1], t[2]]
		for i in 3:
			f.append([m[0][i], m[1][i], m[2][i]])
	else:                                        # robot camera: Rz(yaw) * R
		pos = [x + c * t[0] - s * t[1], y + s * t[0] + c * t[1], z + t[2]]
		for i in 3:
			f.append([c * m[0][i] - s * m[1][i], s * m[0][i] + c * m[1][i], m[2][i]])
	var fwd := gv(f[0])
	var left := gv(f[1])
	var up := gv(f[2])
	cam.transform = Transform3D(Basis(-left, up, -fwd), gv(pos))
	var near := 0.05
	var sx: float = float(W) / float(rig["capture"][0])
	var sy: float = float(H) / float(rig["capture"][1])
	var fx: float = rig["fx"] * sx
	var fy: float = rig["fy"] * sy
	var cx: float = rig["cx"] * sx
	var cy: float = rig["cy"] * sy
	cam.set_frustum(H * near / fy, Vector2((W / 2.0 - cx) * near / fx, (cy - H / 2.0) * near / fy), near, 60.0)
