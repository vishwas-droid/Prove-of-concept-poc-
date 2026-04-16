## fossbot.gd
## Main FOSSBot VehicleBody3D controller.
## Chapters: 7 (modular sensing), 9.1.2 (runtime physics injection),
##           10.1 (visual diagnostics), 5.1.2 (step-sync action application)

extends VehicleBody3D

# ─────────────────────────────────────────────
#  SENSOR NODES  (Chapter 7.1.2)
# ─────────────────────────────────────────────
@onready var sonar_front:  UltrasonicSensor = $Sensors/SonarFront
@onready var sonar_left:   UltrasonicSensor = $Sensors/SonarLeft
@onready var sonar_right:  UltrasonicSensor = $Sensors/SonarRight
@onready var ir_left:      IRSensor         = $Sensors/IRLeft
@onready var ir_center:    IRSensor         = $Sensors/IRCenter
@onready var ir_right:     IRSensor         = $Sensors/IRRight
@onready var imu_sensor:   IMUSensor        = $Sensors/IMU
@onready var odom_sensor:  OdometrySensor   = $Sensors/Odometry

## Wheel nodes
@onready var wheel_fl: VehicleWheel3D = $WheelFL
@onready var wheel_fr: VehicleWheel3D = $WheelFR
@onready var wheel_rl: VehicleWheel3D = $WheelRL
@onready var wheel_rr: VehicleWheel3D = $WheelRR

# ─────────────────────────────────────────────
#  MOTION PARAMETERS
# ─────────────────────────────────────────────
@export var max_engine_force: float = 50.0
@export var max_brake_force: float  = 20.0
@export var max_steer_angle: float  = 0.5   ## radians

# ─────────────────────────────────────────────
#  DIAGNOSTICS  (Chapter 10)
# ─────────────────────────────────────────────
@export var show_planned_path: bool = true
@export var show_local_intent: bool = true
var _planned_path: PackedVector3Array = []
var _intent_path: PackedVector3Array  = []
var _state_label: String = "Idle"           ## State-machine string for HUDD

## Path line meshes
var _path_line: ImmediateMesh
var _intent_line: ImmediateMesh

# ─────────────────────────────────────────────
#  AGGREGATED STATE PACKET (Chapter 5.1.1)
# ─────────────────────────────────────────────
var _last_state_packet: Dictionary = {}

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	## Link body reference to IMU + Odometry
	imu_sensor.link_body(self)
	odom_sensor.link_body(self)

	## Create debug line meshes (Chapter 10.1.2)
	_path_line   = ImmediateMesh.new()
	_intent_line = ImmediateMesh.new()
	var mi1 := MeshInstance3D.new()
	var mi2 := MeshInstance3D.new()
	mi1.mesh = _path_line
	mi2.mesh = _intent_line
	get_tree().root.add_child(mi1)
	get_tree().root.add_child(mi2)

func _physics_process(delta: float) -> void:
	SimInfo.simulation_time += delta

	## Terrain-aware physics injection (Chapter 9.1.2)
	_apply_terrain_physics()

	## Aggregate state packet (Chapter 5.1.1)
	_last_state_packet = _build_state_packet()

	## Visual diagnostics update (Chapter 10)
	_draw_path_overlays()

# ─────────────────────────────────────────────
#  ACTION API  (Chapter 5 — called by FossHandler)
# ─────────────────────────────────────────────

## Apply a normalised action dict: { "throttle": float, "steer": float, "brake": float }
func apply_action(action: Dictionary) -> void:
	var throttle: float = clampf(action.get("throttle", 0.0), -1.0, 1.0)
	var steer: float    = clampf(action.get("steer",    0.0), -1.0, 1.0)
	var brake: float    = clampf(action.get("brake",    0.0),  0.0, 1.0)

	engine_force = throttle * max_engine_force
	steering     = steer    * max_steer_angle
	brake        = brake    * max_brake_force

## Set motor speed directly (m/s → engine force conversion).
func set_motor_speed(left_speed: float, right_speed: float) -> void:
	var avg := (left_speed + right_speed) * 0.5
	var diff := right_speed - left_speed
	apply_action({ "throttle": avg / 2.0, "steer": diff * 0.5, "brake": 0.0 })

## Convenience wrappers matching FossBot Python API
func move_forward(speed: float = 1.0)  -> void: apply_action({"throttle":  speed, "steer": 0.0, "brake": 0.0})
func move_backward(speed: float = 1.0) -> void: apply_action({"throttle": -speed, "steer": 0.0, "brake": 0.0})
func turn_left(speed: float = 0.5)     -> void: apply_action({"throttle":  0.3,  "steer": -speed, "brake": 0.0})
func turn_right(speed: float = 0.5)    -> void: apply_action({"throttle":  0.3,  "steer":  speed, "brake": 0.0})
func stop()                             -> void: apply_action({"throttle":  0.0,  "steer": 0.0, "brake": 1.0})

# ─────────────────────────────────────────────
#  STATE PACKET (Chapter 5.1.1)
# ─────────────────────────────────────────────
func get_state_packet() -> Dictionary:
	return _last_state_packet

func _build_state_packet() -> Dictionary:
	return {
		"step":       SimInfo.current_step,
		"timestamp":  SimInfo.simulation_time,
		"pose": {
			"x":         global_position.x,
			"y":         global_position.y,
			"z":         global_position.z,
			"yaw_deg":   rad_to_deg(global_transform.basis.get_euler().y)
		},
		"sensors": {
			"sonar_front":  sonar_front.get_reading(),
			"sonar_left":   sonar_left.get_reading(),
			"sonar_right":  sonar_right.get_reading(),
			"ir_left":      ir_left.get_reading(),
			"ir_center":    ir_center.get_reading(),
			"ir_right":     ir_right.get_reading(),
			"imu":          imu_sensor.get_reading(),
			"odometry":     odom_sensor.get_reading()
		},
		"terrain":   SimInfo.get_terrain_at(global_position),
		"state_label": _state_label
	}

# ─────────────────────────────────────────────
#  TERRAIN PHYSICS INJECTION  (Chapter 9.1.2)
# ─────────────────────────────────────────────
func _apply_terrain_physics() -> void:
	var mat: Dictionary = SimInfo.get_terrain_at(global_position)
	var friction: float = mat.get("friction", 1.0)
	var traction: float = mat.get("traction", 1.0)

	## Adjust all wheel friction and suspension stiffness
	for wheel in [wheel_fl, wheel_fr, wheel_rl, wheel_rr]:
		if wheel == null:
			continue
		wheel.wheel_friction_slip = friction * traction
		## Rolling resistance via damping approximation
		var rr: float = mat.get("rolling_resistance", 0.01)
		wheel.damping_compression = rr * 10.0

# ─────────────────────────────────────────────
#  VISUAL DIAGNOSTICS  (Chapter 10)
# ─────────────────────────────────────────────

## Set planned path for overlay (call from Python via FossHandler).
func set_planned_path(path: Array) -> void:
	_planned_path.clear()
	for p in path:
		_planned_path.append(Vector3(p[0], p[1] if p.size() > 1 else 0.1, p[2] if p.size() > 2 else p[1]))

## Set short-term local intent path.
func set_intent_path(path: Array) -> void:
	_intent_path.clear()
	for p in path:
		_intent_path.append(Vector3(p[0], p[1] if p.size() > 1 else 0.1, p[2] if p.size() > 2 else p[1]))

func set_state_label(label: String) -> void:
	_state_label = label

func _draw_path_overlays() -> void:
	_draw_line_mesh(_path_line,   _planned_path, Color(0.0, 0.6, 1.0), show_planned_path)
	_draw_line_mesh(_intent_line, _intent_path,  Color(1.0, 0.8, 0.0), show_local_intent)

func _draw_line_mesh(mesh: ImmediateMesh, path: PackedVector3Array, colour: Color, visible: bool) -> void:
	mesh.clear_surfaces()
	if not visible or path.size() < 2:
		return
	mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	mesh.surface_set_color(colour)
	for pt in path:
		mesh.surface_add_vertex(pt)
	mesh.surface_end()

# ─────────────────────────────────────────────
#  RESET  (called on episode start)
# ─────────────────────────────────────────────
func reset_robot(spawn_pose: Dictionary) -> void:
	global_position = Vector3(
		spawn_pose.get("x", 0.0),
		spawn_pose.get("y", 0.1),
		spawn_pose.get("z", 0.0)
	)
	var yaw: float = deg_to_rad(spawn_pose.get("yaw_deg", 0.0))
	global_transform.basis = Basis(Vector3.UP, yaw)
	linear_velocity  = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	stop()
	odom_sensor.reset()
	SimInfo.reset_noise_state()
	SimInfo.current_step = 0
	SimInfo.simulation_time = 0.0
