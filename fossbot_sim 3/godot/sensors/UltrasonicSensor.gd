## UltrasonicSensor.gd
## Raycast-based ultrasonic distance sensor with Gaussian noise injection.
## Chapter 7.1.2 (node-per-sensor) + Chapter 8.2.2 (Gaussian noise)
## Chapter 10.1.1 (ray-cast visualisation)

class_name UltrasonicSensor
extends SensorBase

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
@export var max_range: float = 4.0          ## metres
@export var min_range: float = 0.02
@export var fov_degrees: float = 15.0       ## half-angle cone
@export var ray_count: int = 5              ## rays spread across FOV
@export var debug_draw: bool = true         ## Chapter 10 visual overlay

# ─────────────────────────────────────────────
#  INTERNAL
# ─────────────────────────────────────────────
var _raycasts: Array[RayCast3D] = []
var _debug_lines: Array = []                ## ImmediateMesh instances

func sensor_init() -> void:
	sensor_type = "ultrasonic"
	_setup_raycasts()

func _setup_raycasts() -> void:
	for i in range(ray_count):
		var rc := RayCast3D.new()
		var angle := lerp(-fov_degrees, fov_degrees, float(i) / max(ray_count - 1, 1))
		var dir := Vector3(sin(deg_to_rad(angle)), 0.0, -cos(deg_to_rad(angle)))
		rc.target_position = dir * max_range
		rc.enabled = true
		add_child(rc)
		_raycasts.append(rc)

func sensor_update(delta: float) -> void:
	var min_dist := max_range
	var hit_any := false

	for rc in _raycasts:
		if rc.is_colliding():
			var d := global_position.distance_to(rc.get_collision_point())
			d = clampf(d, min_range, max_range)
			d = SimInfo.apply_noise("ultrasonic", d, delta)
			min_dist = min(min_dist, d)
			hit_any = true

	_last_reading = {
		"sensor_id": sensor_id,
		"type": "ultrasonic",
		"distance_m": min_dist if hit_any else max_range,
		"hit": hit_any,
		"timestamp": SimInfo.simulation_time
	}

	if debug_draw:
		_update_debug_visuals()

func _update_debug_visuals() -> void:
	## Draw colour-coded rays (green→red based on proximity). Chapter 10.1.1
	for i in range(_raycasts.size()):
		var rc: RayCast3D = _raycasts[i]
		var end_pos: Vector3
		if rc.is_colliding():
			end_pos = rc.get_collision_point()
		else:
			end_pos = global_position + rc.target_position

		var dist: float = global_position.distance_to(end_pos)
		var t: float = 1.0 - clampf(dist / max_range, 0.0, 1.0)
		var colour := Color(t, 1.0 - t, 0.0)   # green=far, red=close

		## Use DebugDraw3D plugin pattern or ImmediateMesh if available.
		## This stub calls the global debug helper if present.
		if Engine.has_singleton("DebugDraw"):
			Engine.get_singleton("DebugDraw").draw_line(
				global_position, end_pos, colour
			)
