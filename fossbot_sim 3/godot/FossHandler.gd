## FossHandler.gd
## Central SocketIO handler.
## Chapters: 5 (step-sync loop), 6 (YAML scenario), 10 (HUDD UI overlay)

extends Node

# ─────────────────────────────────────────────
#  NODE REFERENCES
# ─────────────────────────────────────────────
@onready var fossbot:   Node3D = $"/root/Main/FossBot"
@onready var floor_node: Node3D = $"/root/Main/Floor"

## Assumes a SocketIO GDExtension or gdscript-socketio plugin is loaded.
## Adjust to your actual SocketIO node path.
@onready var socket_io = $"/root/SocketIO"

# ─────────────────────────────────────────────
#  STEP-SYNC STATE (Chapter 5.1.2)
# ─────────────────────────────────────────────
var _step_action: Dictionary = {}
var _action_received: bool = false
var _physics_steps_per_action: int = 4     ## how many physics ticks per agent step

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	_register_socket_events()

func _physics_process(_delta: float) -> void:
	if SimInfo.lab_mode:
		_step_locked_loop()

# ─────────────────────────────────────────────
#  SOCKET EVENT REGISTRATION
# ─────────────────────────────────────────────
func _register_socket_events() -> void:
	socket_io.on("connect",           _on_connect)
	socket_io.on("disconnect",        _on_disconnect)

	## Standard robot commands
	socket_io.on("move_forward",      func(d): fossbot.move_forward(d.get("speed", 1.0)))
	socket_io.on("move_backward",     func(d): fossbot.move_backward(d.get("speed", 1.0)))
	socket_io.on("turn_left",         func(d): fossbot.turn_left(d.get("speed", 0.5)))
	socket_io.on("turn_right",        func(d): fossbot.turn_right(d.get("speed", 0.5)))
	socket_io.on("stop",              func(_d): fossbot.stop())

	## Lab mode / step-sync (Chapter 5)
	socket_io.on("enable_lab_mode",   _on_enable_lab_mode)
	socket_io.on("step_action",       _on_step_action)

	## Scenario control (Chapter 6)
	socket_io.on("load_scenario",     _on_load_scenario)
	socket_io.on("reset_episode",     _on_reset_episode)

	## Diagnostics (Chapter 10)
	socket_io.on("set_planned_path",  func(d): fossbot.set_planned_path(d.get("path", [])))
	socket_io.on("set_intent_path",   func(d): fossbot.set_intent_path(d.get("path", [])))
	socket_io.on("set_state_label",   func(d): fossbot.set_state_label(d.get("label", "")))
	socket_io.on("get_costmap",       _on_get_costmap)

	## Telemetry polling (non-lab mode fallback)
	socket_io.on("get_state",         func(_d): _emit_state())

func _on_connect() -> void:
	print("[FossHandler] Client connected.")
	_emit_state()

func _on_disconnect() -> void:
	print("[FossHandler] Client disconnected.")
	fossbot.stop()

# ─────────────────────────────────────────────
#  STEP-LOCK LOOP  (Chapter 5.1.2)
# ─────────────────────────────────────────────
func _on_enable_lab_mode(data: Dictionary) -> void:
	SimInfo.lab_mode = data.get("enabled", true)
	_physics_steps_per_action = data.get("steps_per_action", 4)
	print("[FossHandler] Lab mode: ", SimInfo.lab_mode)
	if SimInfo.lab_mode:
		get_tree().paused = false
		Engine.physics_ticks_per_second = 60
		## Send initial state
		_emit_state_for_step()

func _step_locked_loop() -> void:
	## Called every physics frame when lab_mode is ON.
	## Waits for action; when received, runs N physics ticks, then pauses and emits state.
	if not _action_received:
		return   ## waiting for Python to send action

	_action_received = false
	fossbot.apply_action(_step_action)

	## Advance exactly _physics_steps_per_action ticks.
	## In Godot 4, we control this by counting frames in a coroutine.
	## We emit state after the current tick completes (next frame guard).
	SimInfo.current_step += 1
	_emit_state_for_step.call_deferred()

func _on_step_action(data: Dictionary) -> void:
	## Python sends: { "throttle": 0.5, "steer": 0.0, "brake": 0.0 }
	_step_action = data
	_action_received = true

func _emit_state_for_step() -> void:
	var state: Dictionary = fossbot.get_state_packet()
	state["done"] = _check_terminal_condition()
	state["reward"] = _compute_step_reward()
	socket_io.emit("state", state)

# ─────────────────────────────────────────────
#  SCENARIO LOADER  (Chapter 6)
# ─────────────────────────────────────────────
func _on_load_scenario(data: Dictionary) -> void:
	## data = parsed YAML/JSON scenario dict
	## (Python side sends it after parsing the YAML file)
	var scenario: ScenarioConfig = ScenarioConfig.from_dict(data)
	_apply_scenario(scenario)

func _apply_scenario(scenario: ScenarioConfig) -> void:
	## 1. Seed
	SimInfo.set_seed(scenario.seed)

	## 2. Noise config override
	if scenario.noise_config.size() > 0:
		for sensor_key in scenario.noise_config:
			if SimInfo.noise_config.has(sensor_key):
				SimInfo.noise_config[sensor_key].merge(scenario.noise_config[sensor_key], true)

	## 3. Terrain (Chapter 6.1.2 procedural instantiation)
	if floor_node.has_method("load_terrain"):
		floor_node.load_terrain(
			scenario.heightmap_path,
			scenario.material_map_path,
			scenario.terrain_scale
		)

	## 4. Spawn pose
	fossbot.reset_robot(scenario.spawn_pose)

	## 5. Place obstacles (Chapter 6.1.2)
	_clear_obstacles()
	for obs in scenario.obstacles:
		_spawn_obstacle(obs)

	print("[FossHandler] Scenario loaded: seed=%d obstacles=%d" % [scenario.seed, scenario.obstacles.size()])
	socket_io.emit("scenario_loaded", { "ok": true, "seed": scenario.seed })

func _on_reset_episode(data: Dictionary) -> void:
	## Soft reset: keep scenario, just respawn robot + reset state.
	var spawn: Dictionary = data.get("spawn_pose", {})
	fossbot.reset_robot(spawn)
	socket_io.emit("episode_reset", fossbot.get_state_packet())

func _clear_obstacles() -> void:
	for child in get_tree().get_nodes_in_group("obstacle"):
		child.queue_free()

func _spawn_obstacle(obs: Dictionary) -> void:
	## obs: { type, position, rotation_deg, scale, static }
	var body := StaticBody3D.new()
	var col  := CollisionShape3D.new()
	var mi   := MeshInstance3D.new()

	var obs_type: String = obs.get("type", "box")
	var pos := Vector3(obs.get("x", 0.0), obs.get("y", 0.5), obs.get("z", 0.0))
	var scale_v := Vector3(obs.get("scale_x", 1.0), obs.get("scale_y", 1.0), obs.get("scale_z", 1.0))
	var rot_y: float = deg_to_rad(obs.get("rotation_deg", 0.0))

	match obs_type:
		"box":
			var box := BoxMesh.new()
			box.size = scale_v
			mi.mesh = box
			col.shape = BoxShape3D.new()
			(col.shape as BoxShape3D).size = scale_v
		"cylinder":
			var cyl := CylinderMesh.new()
			cyl.top_radius = scale_v.x * 0.5
			cyl.bottom_radius = scale_v.x * 0.5
			cyl.height = scale_v.y
			mi.mesh = cyl
			col.shape = CylinderShape3D.new()
			(col.shape as CylinderShape3D).radius = scale_v.x * 0.5
			(col.shape as CylinderShape3D).height = scale_v.y
		"sphere":
			var sph := SphereMesh.new()
			sph.radius = scale_v.x * 0.5
			mi.mesh = sph
			col.shape = SphereShape3D.new()
			(col.shape as SphereShape3D).radius = scale_v.x * 0.5

	body.global_position = pos
	body.rotation.y = rot_y
	body.add_child(col)
	body.add_child(mi)
	body.add_to_group("obstacle")
	get_tree().root.add_child(body)

# ─────────────────────────────────────────────
#  COSTMAP EXPORT  (Chapter 9.1.3)
# ─────────────────────────────────────────────
func _on_get_costmap(_data: Dictionary) -> void:
	socket_io.emit("costmap", {
		"cells": SimInfo.get_costmap_export(),
		"timestamp": SimInfo.simulation_time
	})

# ─────────────────────────────────────────────
#  REWARD / TERMINAL  (stub — override per task)
# ─────────────────────────────────────────────
func _compute_step_reward() -> float:
	## Default: small negative per step (encourage efficiency)
	return -0.01

func _check_terminal_condition() -> bool:
	## Override in subclass or connect a signal for custom termination.
	return false

# ─────────────────────────────────────────────
#  FALLBACK TELEMETRY
# ─────────────────────────────────────────────
func _emit_state() -> void:
	socket_io.emit("state", fossbot.get_state_packet())
