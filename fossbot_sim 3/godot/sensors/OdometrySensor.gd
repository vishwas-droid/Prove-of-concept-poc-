## OdometrySensor.gd
## Wheel-encoder based odometry sensor with Gaussian noise.
## Chapter 7.1.2 + Chapter 8.2.2

class_name OdometrySensor
extends SensorBase

var _robot_body: Node3D = null
var _prev_pos: Vector3 = Vector3.ZERO
var _prev_yaw: float = 0.0
var _accum_x: float = 0.0
var _accum_y: float = 0.0
var _accum_theta: float = 0.0

func sensor_init() -> void:
	sensor_type = "odometry"

func link_body(body: Node3D) -> void:
	_robot_body = body
	_prev_pos = body.global_position
	_prev_yaw = body.global_transform.basis.get_euler().y

func reset() -> void:
	_accum_x = 0.0
	_accum_y = 0.0
	_accum_theta = 0.0
	if _robot_body:
		_prev_pos = _robot_body.global_position
		_prev_yaw = _robot_body.global_transform.basis.get_euler().y

func sensor_update(delta: float) -> void:
	if _robot_body == null:
		return

	var cur_pos: Vector3 = _robot_body.global_position
	var cur_yaw: float = _robot_body.global_transform.basis.get_euler().y

	var dx: float = cur_pos.x - _prev_pos.x
	var dz: float = cur_pos.z - _prev_pos.z
	var dtheta: float = cur_yaw - _prev_yaw

	## Apply Gaussian noise to incremental motion (Chapter 8.2.2)
	dx = SimInfo.apply_noise("odometry", dx, delta)
	dz = SimInfo.apply_noise("odometry", dz, delta)

	_accum_x += dx
	_accum_y += dz
	_accum_theta += dtheta

	_prev_pos = cur_pos
	_prev_yaw = cur_yaw

	_last_reading = {
		"sensor_id": sensor_id,
		"type": "odometry",
		"pose": {
			"x": _accum_x,
			"y": _accum_y,
			"theta_deg": rad_to_deg(_accum_theta)
		},
		"velocity": {
			"x": dx / max(delta, 1e-6),
			"z": dz / max(delta, 1e-6)
		},
		"timestamp": SimInfo.simulation_time
	}
