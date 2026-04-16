## SensorBase.gd
## Abstract base class for all FOSSBot sensors.
## All sensor nodes must extend this class.
## Chapter 7.1.1 — Sensor Base Class

class_name SensorBase
extends Node

# ─────────────────────────────────────────────
#  METADATA
# ─────────────────────────────────────────────
@export var sensor_id: String = "sensor"
@export var sensor_type: String = "base"        ## "ultrasonic" | "ir" | "imu" | "odometry"
@export var enabled: bool = true
@export var update_rate_hz: float = 30.0        ## How often sensor ticks (Hz)

# ─────────────────────────────────────────────
#  INTERNAL
# ─────────────────────────────────────────────
var _last_reading: Dictionary = {}
var _time_since_update: float = 0.0

signal reading_ready(sensor_id: String, data: Dictionary)

# ─────────────────────────────────────────────
#  VIRTUAL API  (override in subclasses)
# ─────────────────────────────────────────────

## Called once when sensor initialises.
func sensor_init() -> void:
	pass

## Called every physics tick. Subclass must populate _last_reading.
func sensor_update(_delta: float) -> void:
	pass

## Returns structured JSON-serialisable dict (Chapter 7.1.3).
func get_reading() -> Dictionary:
	return _last_reading.duplicate()

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	sensor_init()

func _physics_process(delta: float) -> void:
	if not enabled:
		return
	_time_since_update += delta
	if _time_since_update >= (1.0 / update_rate_hz):
		_time_since_update = 0.0
		sensor_update(delta)
		emit_signal("reading_ready", sensor_id, _last_reading)
