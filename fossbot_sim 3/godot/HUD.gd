## HUD.gd
## Real-Time Heads-Up Diagnostic Display (HUDD) — Chapter 10.1.3
## Attach to a CanvasLayer node in your Main scene.
## Displays: state-machine label, sensor readings, step counter, costmap toggle.

extends CanvasLayer

# ─────────────────────────────────────────────
#  UI REFERENCES
# ─────────────────────────────────────────────
@onready var lbl_state:     Label = $Panel/VBox/LblState
@onready var lbl_step:      Label = $Panel/VBox/LblStep
@onready var lbl_sonar:     Label = $Panel/VBox/LblSonar
@onready var lbl_ir:        Label = $Panel/VBox/LblIR
@onready var lbl_odom:      Label = $Panel/VBox/LblOdom
@onready var lbl_terrain:   Label = $Panel/VBox/LblTerrain
@onready var btn_costmap:   Button = $Panel/VBox/BtnCostmap
@onready var btn_paths:     Button = $Panel/VBox/BtnPaths
@onready var costmap_overlay: Control = $CostmapOverlay

@onready var fossbot: Node3D = $"/root/Main/FossBot"

var _show_costmap: bool = false
var _show_paths: bool = true
var _costmap_texture: ImageTexture = null

# ─────────────────────────────────────────────
#  LIFECYCLE
# ─────────────────────────────────────────────
func _ready() -> void:
	btn_costmap.pressed.connect(_toggle_costmap)
	btn_paths.pressed.connect(_toggle_paths)
	costmap_overlay.visible = false

func _process(_delta: float) -> void:
	if not fossbot:
		return
	var state: Dictionary = fossbot.get_state_packet()
	_update_labels(state)

func _update_labels(state: Dictionary) -> void:
	lbl_state.text   = "State: " + state.get("state_label", "—")
	lbl_step.text    = "Step: %d  |  T: %.2f s" % [state.get("step", 0), state.get("timestamp", 0.0)]

	var sensors: Dictionary = state.get("sensors", {})

	## Sonar
	var sf: Dictionary = sensors.get("sonar_front", {})
	var sl: Dictionary = sensors.get("sonar_left",  {})
	var sr: Dictionary = sensors.get("sonar_right", {})
	lbl_sonar.text = "Sonar  F:%.2fm  L:%.2fm  R:%.2fm" % [
		sf.get("distance_m", 0.0), sl.get("distance_m", 0.0), sr.get("distance_m", 0.0)
	]

	## IR
	var irl: Dictionary = sensors.get("ir_left",   {})
	var irc: Dictionary = sensors.get("ir_center", {})
	var irr: Dictionary = sensors.get("ir_right",  {})
	lbl_ir.text = "IR  L:%s  C:%s  R:%s" % [
		_bool_str(irl.get("detected", false)),
		_bool_str(irc.get("detected", false)),
		_bool_str(irr.get("detected", false))
	]

	## Odometry
	var odom: Dictionary = sensors.get("odometry", {}).get("pose", {})
	lbl_odom.text = "Odom  x:%.2f  y:%.2f  θ:%.1f°" % [
		odom.get("x", 0.0), odom.get("y", 0.0), odom.get("theta_deg", 0.0)
	]

	## Terrain
	var terrain: Dictionary = state.get("terrain", {})
	lbl_terrain.text = "Terrain: %s  cost:%.1f  µ:%.2f" % [
		terrain.get("material","—"), terrain.get("cost",1.0), terrain.get("friction",1.0)
	]

func _bool_str(v: bool) -> String:
	return "●" if v else "○"

# ─────────────────────────────────────────────
#  COSTMAP OVERLAY  (Chapter 10.1.3)
# ─────────────────────────────────────────────
func _toggle_costmap() -> void:
	_show_costmap = not _show_costmap
	costmap_overlay.visible = _show_costmap
	btn_costmap.text = ("Hide" if _show_costmap else "Show") + " Costmap"
	if _show_costmap:
		_render_costmap_texture()

func _render_costmap_texture() -> void:
	## Build an Image from SimInfo.terrain_costmap and display as texture.
	var cells: Array = SimInfo.get_costmap_export()
	if cells.is_empty():
		return

	## Find bounds
	var min_x := INF; var max_x := -INF
	var min_z := INF; var max_z := -INF
	for c in cells:
		min_x = min(min_x, c["cell_x"]); max_x = max(max_x, c["cell_x"])
		min_z = min(min_z, c["cell_y"]); max_z = max(max_z, c["cell_y"])

	var w: int = max(int(max_x - min_x) + 1, 1)
	var h: int = max(int(max_z - min_z) + 1, 1)
	var img := Image.create(w, h, false, Image.FORMAT_RGBA8)

	for c in cells:
		var px: int = int(c["cell_x"] - min_x)
		var pz: int = int(c["cell_y"] - min_z)
		var cost: float = clampf(c.get("cost", 1.0) / 3.0, 0.0, 1.0)  # normalise 0-3
		## Green = low cost, red = high cost
		var col := Color(cost, 1.0 - cost, 0.0, 0.7)
		img.set_pixel(px, pz, col)

	_costmap_texture = ImageTexture.create_from_image(img)
	var tex_rect: TextureRect = costmap_overlay.get_node_or_null("TextureRect")
	if tex_rect:
		tex_rect.texture = _costmap_texture

func _toggle_paths() -> void:
	_show_paths = not _show_paths
	if fossbot:
		fossbot.show_planned_path = _show_paths
		fossbot.show_local_intent = _show_paths
	btn_paths.text = ("Hide" if _show_paths else "Show") + " Paths"
