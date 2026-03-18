## IMUSensor.gd
## Inertial Measurement Unit sensor with drift + bias noise model.
## Chapter 7.1.2 + Chapter 8.2.2 (Drift and Bias / Random Walk)

class_name IMUSensor
extends SensorBase

var _robot_body: Node3D = null   ## Set by fossbot.gd after instantiation

func sensor_init() -> void:
	sensor_type = "imu"

## Called by fossbot.gd to link the VehicleBody.
func link_body(body: Node3D) -> void:
	_robot_body = body

func sensor_update(delta: float) -> void:
	if _robot_body == null:
		return

	var lin_vel: Vector3 = Vector3.ZERO
	var ang_vel: Vector3 = Vector3.ZERO
	var orientation: Basis = Basis.IDENTITY

	## Duck-type: works with RigidBody3D / VehicleBody3D / CharacterBody3D
	if _robot_body.has_method("get_linear_velocity"):
		lin_vel = _robot_body.linear_velocity
	elif "velocity" in _robot_body:
		lin_vel = _robot_body.velocity

	if "angular_velocity" in _robot_body:
		ang_vel = _robot_body.angular_velocity

	orientation = _robot_body.global_transform.basis

	## Apply drift + bias (Chapter 8.2.2)
	var noisy_ang := SimInfo.imu_noise(ang_vel, delta)

	## Gaussian noise on linear acceleration estimation
	var accel_x: float = SimInfo.gaussian_noise(lin_vel.x, 0.01)
	var accel_z: float = SimInfo.gaussian_noise(lin_vel.z, 0.01)

	var euler: Vector3 = orientation.get_euler()

	_last_reading = {
		"sensor_id": sensor_id,
		"type": "imu",
		"linear_velocity": { "x": lin_vel.x, "y": lin_vel.y, "z": lin_vel.z },
		"angular_velocity": { "x": noisy_ang.x, "y": noisy_ang.y, "z": noisy_ang.z },
		"acceleration_estimate": { "x": accel_x, "z": accel_z },
		"orientation": {
			"roll":  rad_to_deg(euler.z),
			"pitch": rad_to_deg(euler.x),
			"yaw":   rad_to_deg(euler.y)
		},
		"timestamp": SimInfo.simulation_time
	}
