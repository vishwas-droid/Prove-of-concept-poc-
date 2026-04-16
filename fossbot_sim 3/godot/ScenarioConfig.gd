## ScenarioConfig.gd
## Data class representing a parsed scenario configuration (Chapter 6.1).
## YAML/JSON deserialization happens on the Python side; Godot receives a dict.

class_name ScenarioConfig
extends RefCounted

# ─────────────────────────────────────────────
#  FIELDS  (map to YAML keys)
# ─────────────────────────────────────────────
var name: String                   = "unnamed"
var seed: int                      = 42
var spawn_pose: Dictionary         = { "x": 0.0, "y": 0.1, "z": 0.0, "yaw_deg": 0.0 }
var goal_coordinates: Dictionary   = { "x": 5.0, "z": 5.0 }
var heightmap_path: String         = ""
var material_map_path: String      = ""
var terrain_scale: Vector3         = Vector3(100.0, 10.0, 100.0)
var floor_texture_map: String      = ""
var obstacles: Array               = []       ## Array of obstacle dicts
var noise_config: Dictionary       = {}       ## Per-sensor noise overrides

# ─────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────
static func from_dict(d: Dictionary) -> ScenarioConfig:
	var cfg := ScenarioConfig.new()
	cfg.name               = d.get("name", "unnamed")
	cfg.seed               = d.get("seed", 42)
	cfg.spawn_pose         = d.get("spawn_pose",       cfg.spawn_pose)
	cfg.goal_coordinates   = d.get("goal_coordinates", cfg.goal_coordinates)
	cfg.heightmap_path     = d.get("heightmap_path",   "")
	cfg.material_map_path  = d.get("material_map_path","")
	cfg.floor_texture_map  = d.get("floor_texture_map","")
	cfg.obstacles          = d.get("obstacles",        [])
	cfg.noise_config       = d.get("noise_config",     {})
	if d.has("terrain_scale"):
		var ts = d["terrain_scale"]
		cfg.terrain_scale  = Vector3(ts.get("x",100.0), ts.get("y",10.0), ts.get("z",100.0))
	return cfg

func to_dict() -> Dictionary:
	return {
		"name":             name,
		"seed":             seed,
		"spawn_pose":       spawn_pose,
		"goal_coordinates": goal_coordinates,
		"heightmap_path":   heightmap_path,
		"material_map_path":material_map_path,
		"floor_texture_map":floor_texture_map,
		"obstacles":        obstacles,
		"noise_config":     noise_config,
		"terrain_scale":    { "x": terrain_scale.x, "y": terrain_scale.y, "z": terrain_scale.z }
	}
