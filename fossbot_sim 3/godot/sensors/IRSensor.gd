## IRSensor.gd
## Ground-facing IR line/obstacle sensor with salt-and-pepper noise.
## Chapter 7.1.2 + Chapter 8.2.2

class_name IRSensor
extends SensorBase

@export var detect_range: float = 0.12      ## metres downward
@export var debug_draw: bool = true

var _raycast: RayCast3D

func sensor_init() -> void:
	sensor_type = "ir"
	_raycast = RayCast3D.new()
	_raycast.target_position = Vector3(0, -detect_range, 0)
	_raycast.enabled = true
	add_child(_raycast)

func sensor_update(delta: float) -> void:
	var raw: float = 1.0 if _raycast.is_colliding() else 0.0
	var noisy: float = SimInfo.apply_noise("ir", raw, delta)
	var detected: bool = noisy > 0.5

	## Read surface colour for line detection if colliding
	var surface_color: Color = Color.WHITE
	if _raycast.is_colliding():
		var collider := _raycast.get_collider()
		if collider and collider.has_method("get_surface_color_at"):
			surface_color = collider.get_surface_color_at(_raycast.get_collision_point())

	_last_reading = {
		"sensor_id": sensor_id,
		"type": "ir",
		"detected": detected,
		"raw": raw,
		"surface_luminance": surface_color.v,
		"timestamp": SimInfo.simulation_time
	}

	if debug_draw and _raycast.is_colliding():
		_draw_ground_trace(_raycast.get_collision_point(), detected)

func _draw_ground_trace(pos: Vector3, active: bool) -> void:
	## Visual indicator on floor (Chapter 10.1.1 Ground-Sensor Traces)
	if Engine.has_singleton("DebugDraw"):
		var col := Color.RED if active else Color.CYAN
		Engine.get_singleton("DebugDraw").draw_sphere(pos, 0.02, col)
