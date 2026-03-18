## sim_info.gd
## Global singleton: seed manager, noise injection layer, shared simulation state.
## Attach as AutoLoad in Project Settings → AutoLoad → SimInfo

extends Node

# ─────────────────────────────────────────────
#  SIMULATION STATE
# ─────────────────────────────────────────────
var lab_mode: bool = false          ## Step-lock enabled (Chapter 5)
var step_pending: bool = false      ## True while waiting for action from client
var current_step: int = 0
var simulation_time: float = 0.0

# ─────────────────────────────────────────────
#  SEED MANAGER  (Chapter 6.1.3 & 8.2.3)
# ─────────────────────────────────────────────
var global_seed: int = 42
var _rng: RandomNumberGenerator = RandomNumberGenerator.new()

func set_seed(seed_value: int) -> void:
	global_seed = seed_value
	_rng.seed = seed_value

func get_rng() -> RandomNumberGenerator:
	return _rng

# ─────────────────────────────────────────────
#  NOISE CONFIGURATION  (Chapter 8)
# ─────────────────────────────────────────────
## Master switch
var noise_enabled: bool = true

## Per-sensor noise parameters (tunable via scenario YAML)
var noise_config: Dictionary = {
	"ultrasonic": {
		"type": "gaussian",
		"stddev": 0.02,       # metres
		"enabled": true
	},
	"ir": {
		"type": "salt_pepper",
		"prob": 0.05,         # 5 % dropout probability
		"enabled": true
	},
	"imu": {
		"type": "drift_bias",
		"drift_rate": 0.001,  # rad/s accumulated error
		"bias": Vector3(0.002, 0.001, 0.0015),
		"enabled": true
	},
	"odometry": {
		"type": "gaussian",
		"stddev": 0.005,
		"enabled": true
	}
}

## IMU drift accumulator
var _imu_drift_accum: Vector3 = Vector3.ZERO

# ─────────────────────────────────────────────
#  NOISE INJECTION API  (Chapter 8.2)
# ─────────────────────────────────────────────

## Apply Gaussian noise to a float value.
func gaussian_noise(value: float, stddev: float) -> float:
	if not noise_enabled:
		return value
	# Box-Muller transform
	var u1 := _rng.randf()
	var u2 := _rng.randf()
	u1 = max(u1, 1e-10)   # avoid log(0)
	var z: float = sqrt(-2.0 * log(u1)) * cos(TAU * u2)
	return value + z * stddev

## Apply salt-and-pepper noise to a boolean/discrete sensor.
## Returns original value most of the time; random 0/1 on dropout.
func salt_pepper_noise(value: float, prob: float) -> float:
	if not noise_enabled:
		return value
	if _rng.randf() < prob:
		return float(_rng.randi() % 2)   # 0 or 1
	return value

## Apply IMU drift/bias noise to a Vector3 angular reading.
func imu_noise(angular_velocity: Vector3, delta: float) -> Vector3:
	if not noise_enabled:
		return angular_velocity
	var cfg: Dictionary = noise_config["imu"]
	if not cfg["enabled"]:
		return angular_velocity
	_imu_drift_accum += Vector3.ONE * cfg["drift_rate"] * delta
	return angular_velocity + cfg["bias"] + _imu_drift_accum

## Dispatch noise by sensor type string.
func apply_noise(sensor_type: String, value: float, delta: float = 0.0) -> float:
	if not noise_enabled:
		return value
	if not noise_config.has(sensor_type):
		return value
	var cfg: Dictionary = noise_config[sensor_type]
	if not cfg.get("enabled", true):
		return value
	match cfg["type"]:
		"gaussian":
			return gaussian_noise(value, cfg["stddev"])
		"salt_pepper":
			return salt_pepper_noise(value, cfg["prob"])
		_:
			return value

## Reset drift accumulators (call on episode reset).
func reset_noise_state() -> void:
	_imu_drift_accum = Vector3.ZERO
	_rng.seed = global_seed   # deterministic replay

# ─────────────────────────────────────────────
#  TERRAIN COSTMAP  (Chapter 9.1.3)
# ─────────────────────────────────────────────
## Populated by floor.gd at runtime.
## Key: Vector2i grid cell  →  Value: { friction, traction, rolling_resistance, cost }
var terrain_costmap: Dictionary = {}

func register_terrain_cell(cell: Vector2i, props: Dictionary) -> void:
	terrain_costmap[cell] = props

func get_terrain_at(world_pos: Vector3) -> Dictionary:
	## Convert world XZ to costmap cell (1 m grid).
	var cell := Vector2i(int(world_pos.x), int(world_pos.z))
	return terrain_costmap.get(cell, _default_terrain())

func _default_terrain() -> Dictionary:
	return { "friction": 1.0, "traction": 1.0, "rolling_resistance": 0.01, "cost": 1.0, "material": "default" }

func get_costmap_export() -> Array:
	## Returns costmap as a list of dicts for the Python API.
	var result: Array = []
	for cell in terrain_costmap:
		var entry: Dictionary = terrain_costmap[cell].duplicate()
		entry["cell_x"] = cell.x
		entry["cell_y"] = cell.y
		result.append(entry)
	return result

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	set_seed(global_seed)
