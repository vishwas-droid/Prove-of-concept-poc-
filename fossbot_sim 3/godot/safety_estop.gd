## safety_estop.gd
## Chapter 15.1.2 — Godot-Side Emergency Stop
## Attach this as a child node of FossBot in the scene tree.
## Monitors sensors every physics tick and overrides commands when critical.

extends Node

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
@export var enabled: bool = true
@export var front_collision_threshold: float = 0.10   ## metres
@export var watchdog_timeout: float = 1.5             ## seconds
@export var heartbeat_event: String = "heartbeat"     ## SocketIO event name

# ─────────────────────────────────────────────
#  REFERENCES
# ─────────────────────────────────────────────
@onready var fossbot: Node = get_parent()
@onready var socket_io = $"/root/SocketIO"

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
var _estop_active: bool = false
var _last_heartbeat: float = 0.0
var _estop_reason: String = ""
var _intervention_count: int = 0

signal estop_triggered(reason: String)
signal estop_cleared()

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	_last_heartbeat = Time.get_ticks_msec() / 1000.0
	if socket_io:
		socket_io.on(heartbeat_event, _on_heartbeat)
	print("[EStop] Safety watchdog active.")

func _physics_process(_delta: float) -> void:
	if not enabled:
		return
	_check_watchdog()
	_check_sensor_conditions()
	if _estop_active:
		_apply_safe_stop()

# ─────────────────────────────────────────────
#  WATCHDOG CHECK  (Chapter 15.1.3)
# ─────────────────────────────────────────────
func _check_watchdog() -> void:
	var now: float = Time.get_ticks_msec() / 1000.0
	var elapsed: float = now - _last_heartbeat
	if elapsed > watchdog_timeout and not _estop_active:
		_activate_estop("watchdog_timeout(%.1fs)" % elapsed)

func _on_heartbeat(_data: Dictionary) -> void:
	_last_heartbeat = Time.get_ticks_msec() / 1000.0
	## Clear watchdog-triggered E-Stop when comms resume
	if _estop_active and _estop_reason.begins_with("watchdog"):
		_clear_estop()

# ─────────────────────────────────────────────
#  SENSOR CONDITION CHECKS  (Chapter 15.1.2)
# ─────────────────────────────────────────────
func _check_sensor_conditions() -> void:
	if not fossbot.has_method("get_state_packet"):
		return

	var state: Dictionary = fossbot.get_state_packet()
	var sensors: Dictionary = state.get("sensors", {})

	## Front collision check
	var sonar_f: float = sensors.get("sonar_front", {}).get("distance_m", 4.0)
	var robot_throttle: float = fossbot.engine_force / max(fossbot.max_engine_force, 1.0)
	if sonar_f < front_collision_threshold and robot_throttle > 0.0:
		_activate_estop("imminent_collision(sonar=%.2fm)" % sonar_f)
		return

	## Cliff detection: all IR sensors off surface
	var ir_l: bool = sensors.get("ir_left",   {}).get("detected", true)
	var ir_c: bool = sensors.get("ir_center", {}).get("detected", true)
	var ir_r: bool = sensors.get("ir_right",  {}).get("detected", true)
	if not ir_l and not ir_c and not ir_r:
		_activate_estop("cliff_detected(all_IR_off)")
		return

	## NaN / Inf in incoming engine force
	if not is_finite(fossbot.engine_force) or not is_finite(fossbot.steering):
		_activate_estop("nan_in_controls")
		fossbot.engine_force = 0.0
		fossbot.steering = 0.0
		return

	## If we get here and E-Stop was active due to sensor condition, clear it
	if _estop_active and not _estop_reason.begins_with("watchdog"):
		_clear_estop()

# ─────────────────────────────────────────────
#  ESTOP ACTIONS
# ─────────────────────────────────────────────
func _apply_safe_stop() -> void:
	## Override robot controls at physics level — bypasses SocketIO entirely
	fossbot.engine_force = 0.0
	fossbot.steering     = 0.0
	fossbot.brake        = fossbot.max_brake_force

func _activate_estop(reason: String) -> void:
	if _estop_active:
		return
	_estop_active = true
	_estop_reason = reason
	_intervention_count += 1
	print("[EStop] TRIGGERED: %s (intervention #%d)" % [reason, _intervention_count])
	emit_signal("estop_triggered", reason)
	## Notify Python client
	if socket_io:
		socket_io.emit("estop_triggered", {
			"reason": reason,
			"step": SimInfo.current_step,
			"timestamp": SimInfo.simulation_time
		})

func _clear_estop() -> void:
	_estop_active = false
	_estop_reason = ""
	print("[EStop] Cleared.")
	emit_signal("estop_cleared")

# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
func is_estop_active() -> bool:
	return _estop_active

func get_intervention_count() -> int:
	return _intervention_count

func manual_estop() -> void:
	_activate_estop("manual_trigger")

func manual_clear() -> void:
	_clear_estop()
