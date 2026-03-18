## floor.gd
## Heightmap terrain loader + Semantic Surface Mapping.
## Chapter 6.1.2 (procedural instantiation) + Chapter 9.1.1/9.1.3 (material layer + costmap)

extends Node3D

# ─────────────────────────────────────────────
#  MATERIAL DEFINITIONS  (Chapter 9.1.1)
# ─────────────────────────────────────────────
## Maps RGBA colour key (from texture) → physics material properties.
## Color tolerance applied during lookup.
const MATERIAL_MAP: Dictionary = {
	"default":  { "friction": 1.00, "traction": 1.00, "rolling_resistance": 0.01, "cost": 1.0 },
	"ice":      { "friction": 0.10, "traction": 0.15, "rolling_resistance": 0.002,"cost": 2.5 },
	"gravel":   { "friction": 0.80, "traction": 0.70, "rolling_resistance": 0.05, "cost": 1.8 },
	"carpet":   { "friction": 1.50, "traction": 1.30, "rolling_resistance": 0.08, "cost": 1.2 },
	"mud":      { "friction": 0.60, "traction": 0.50, "rolling_resistance": 0.15, "cost": 3.0 },
	"grass":    { "friction": 0.90, "traction": 0.80, "rolling_resistance": 0.06, "cost": 1.5 },
	"asphalt":  { "friction": 1.10, "traction": 1.05, "rolling_resistance": 0.01, "cost": 1.0 }
}

## Colour-to-material ID lookup (colour from texture map pixel).
const COLOR_TO_MATERIAL: Dictionary = {
	"0000ff": "ice",       # Blue
	"808080": "gravel",    # Grey
	"ff8000": "carpet",    # Orange
	"5c3d1e": "mud",       # Brown
	"00cc00": "grass",     # Green
	"333333": "asphalt"    # Dark grey
}

# ─────────────────────────────────────────────
#  HEIGHTMAP STATE
# ─────────────────────────────────────────────
var _heightmap_image: Image = null
var _material_image: Image = null       ## separate RGBA material map texture
var _terrain_mesh: MeshInstance3D = null
var _heightmap_scale: Vector3 = Vector3(100, 10, 100)
var _grid_resolution: int = 128

# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

## Load heightmap and optional material map from paths.
## Called by FossHandler.gd during scenario load.
func load_terrain(heightmap_path: String, material_map_path: String = "", scale: Vector3 = Vector3(100, 10, 100)) -> void:
	_heightmap_scale = scale

	if heightmap_path != "":
		_heightmap_image = Image.new()
		var err := _heightmap_image.load(heightmap_path)
		if err != OK:
			push_error("floor.gd: failed to load heightmap: " + heightmap_path)
			_heightmap_image = null

	if material_map_path != "":
		_material_image = Image.new()
		var err2 := _material_image.load(material_map_path)
		if err2 != OK:
			push_warning("floor.gd: no material map loaded, using defaults.")
			_material_image = null

	_build_terrain_mesh()
	_build_costmap()

## Returns surface colour at world position (used by IRSensor).
func get_surface_color_at(world_pos: Vector3) -> Color:
	if _material_image == null:
		return Color.WHITE
	var uv := _world_to_uv(world_pos)
	_material_image.decompress()
	return _material_image.get_pixelv(Vector2i(
		int(uv.x * _material_image.get_width()),
		int(uv.y * _material_image.get_height())
	))

## Returns physics material for a given world position.
func get_material_at(world_pos: Vector3) -> Dictionary:
	if _material_image == null:
		return MATERIAL_MAP["default"]
	var col: Color = get_surface_color_at(world_pos)
	var hex: String = _color_to_hex(col)
	var mat_id: String = COLOR_TO_MATERIAL.get(hex, "default")
	var mat: Dictionary = MATERIAL_MAP.get(mat_id, MATERIAL_MAP["default"]).duplicate()
	mat["material"] = mat_id
	return mat

# ─────────────────────────────────────────────
#  INTERNAL
# ─────────────────────────────────────────────

func _build_terrain_mesh() -> void:
	if _terrain_mesh:
		_terrain_mesh.queue_free()

	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var res := _grid_resolution
	for z in range(res):
		for x in range(res):
			var verts: Array[Vector3] = []
			for dz in [0, 1]:
				for dx in [0, 1]:
					var nx := x + dx
					var nz := z + dz
					var h := _sample_height(nx, nz, res)
					verts.append(Vector3(
						(float(nx) / res) * _heightmap_scale.x - _heightmap_scale.x * 0.5,
						h * _heightmap_scale.y,
						(float(nz) / res) * _heightmap_scale.z - _heightmap_scale.z * 0.5
					))
			## Two triangles per quad
			for idx in [0, 2, 1, 1, 2, 3]:
				st.add_vertex(verts[idx])

	var mesh: ArrayMesh = st.commit()
	_terrain_mesh = MeshInstance3D.new()
	_terrain_mesh.mesh = mesh

	## Add static body for collision
	var static_body := StaticBody3D.new()
	var collision := CollisionShape3D.new()
	var shape := ConcavePolygonShape3D.new()
	shape.set_faces(mesh.get_faces())
	collision.shape = shape
	static_body.add_child(collision)
	_terrain_mesh.add_child(static_body)
	add_child(_terrain_mesh)

func _build_costmap() -> void:
	## Populate SimInfo.terrain_costmap (Chapter 9.1.3)
	SimInfo.terrain_costmap.clear()
	if _material_image == null:
		return
	var img_w: int = _material_image.get_width()
	var img_h: int = _material_image.get_height()
	_material_image.decompress()
	for iz in range(img_h):
		for ix in range(img_w):
			var col: Color = _material_image.get_pixel(ix, iz)
			var hex: String = _color_to_hex(col)
			var mat_id: String = COLOR_TO_MATERIAL.get(hex, "default")
			var props: Dictionary = MATERIAL_MAP.get(mat_id, MATERIAL_MAP["default"]).duplicate()
			props["material"] = mat_id
			## Map image pixel to world cell
			var wx: int = int((float(ix) / img_w) * _heightmap_scale.x - _heightmap_scale.x * 0.5)
			var wz: int = int((float(iz) / img_h) * _heightmap_scale.z - _heightmap_scale.z * 0.5)
			SimInfo.register_terrain_cell(Vector2i(wx, wz), props)

func _sample_height(x: int, z: int, res: int) -> float:
	if _heightmap_image == null:
		return 0.0
	_heightmap_image.decompress()
	var px: int = int(float(x) / res * _heightmap_image.get_width())
	var pz: int = int(float(z) / res * _heightmap_image.get_height())
	px = clampi(px, 0, _heightmap_image.get_width() - 1)
	pz = clampi(pz, 0, _heightmap_image.get_height() - 1)
	return _heightmap_image.get_pixel(px, pz).r

func _world_to_uv(world_pos: Vector3) -> Vector2:
	var u: float = (world_pos.x + _heightmap_scale.x * 0.5) / _heightmap_scale.x
	var v: float = (world_pos.z + _heightmap_scale.z * 0.5) / _heightmap_scale.z
	return Vector2(clampf(u, 0.0, 1.0), clampf(v, 0.0, 1.0))

func _color_to_hex(c: Color) -> String:
	return "%02x%02x%02x" % [int(c.r * 255), int(c.g * 255), int(c.b * 255)]
