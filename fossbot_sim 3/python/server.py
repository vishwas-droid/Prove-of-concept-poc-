"""
server.py
=========
Flask-SocketIO relay server.
Acts as the central message broker between the Godot simulator
and any connected Python client (direct scripts, FOSSBot-Gym, etc.).

Architecture:
    Godot  <──SocketIO──>  server.py  <──SocketIO──>  Python Agent

Run:
    pip install flask flask-socketio eventlet
    python server.py [--host 0.0.0.0] [--port 5000] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, Optional

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "fossbot_secret"

sio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("fossbot.server")

# ─────────────────────────────────────────────
#  SESSION REGISTRY
# ─────────────────────────────────────────────
# We support exactly one Godot instance and N Python clients.
# Godot joins room "godot"; Python clients join room "clients".

_godot_sid: Optional[str] = None       # Godot simulator session id
_client_sids: set[str] = set()         # Python agent session ids
_latest_state: Dict[str, Any] = {}     # Last state packet from Godot
_server_start: float = time.time()


def _uptime() -> str:
    s = int(time.time() - _server_start)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


# ─────────────────────────────────────────────
#  CONNECTION LIFECYCLE
# ─────────────────────────────────────────────

@sio.on("connect")
def on_connect():
    sid = request.sid
    role = request.args.get("role", "client")   # ?role=godot or ?role=client

    if role == "godot":
        global _godot_sid
        _godot_sid = sid
        join_room("godot", sid=sid)
        log.info(f"[{_uptime()}] Godot simulator connected  sid={sid[:8]}")
        emit("server_ready", {"message": "Godot registered."}, to=sid)
    else:
        _client_sids.add(sid)
        join_room("clients", sid=sid)
        log.info(f"[{_uptime()}] Python client connected  sid={sid[:8]}  total={len(_client_sids)}")
        # Send last known state immediately so client doesn't have to poll
        if _latest_state:
            emit("state", _latest_state, to=sid)
        emit("server_ready", {"message": "Client registered."}, to=sid)


@sio.on("disconnect")
def on_disconnect():
    global _godot_sid
    sid = request.sid
    if sid == _godot_sid:
        _godot_sid = None
        log.warning(f"[{_uptime()}] Godot simulator DISCONNECTED.")
        sio.emit("godot_disconnected", {}, to="clients")
    elif sid in _client_sids:
        _client_sids.discard(sid)
        log.info(f"[{_uptime()}] Python client disconnected  sid={sid[:8]}  remaining={len(_client_sids)}")


# ─────────────────────────────────────────────
#  GODOT → CLIENTS  (simulator pushes state)
# ─────────────────────────────────────────────

@sio.on("state")
def relay_state(data: Dict):
    """Godot emits 'state'; relay to all Python clients."""
    global _latest_state
    _latest_state = data
    sio.emit("state", data, to="clients")


@sio.on("scenario_loaded")
def relay_scenario_loaded(data: Dict):
    sio.emit("scenario_loaded", data, to="clients")


@sio.on("episode_reset")
def relay_episode_reset(data: Dict):
    global _latest_state
    _latest_state = data
    sio.emit("episode_reset", data, to="clients")


@sio.on("costmap")
def relay_costmap(data: Dict):
    sio.emit("costmap", data, to="clients")


# ─────────────────────────────────────────────
#  CLIENTS → GODOT  (agent sends commands)
# ─────────────────────────────────────────────

def _to_godot(event: str, data: Dict) -> None:
    """Forward an event to Godot. Warns if Godot is not connected."""
    if _godot_sid is None:
        log.warning(f"[{_uptime()}] DROP '{event}': Godot not connected.")
        return
    sio.emit(event, data, to=_godot_sid)


# ── Robot motion commands
@sio.on("move_forward")
def fwd(d):   _to_godot("move_forward",  d)

@sio.on("move_backward")
def bwd(d):   _to_godot("move_backward", d)

@sio.on("turn_left")
def tl(d):    _to_godot("turn_left",     d)

@sio.on("turn_right")
def tr(d):    _to_godot("turn_right",    d)

@sio.on("stop")
def stop(d):  _to_godot("stop",          d)

# ── Lab mode / step-sync
@sio.on("enable_lab_mode")
def lab(d):         _to_godot("enable_lab_mode", d)

@sio.on("step_action")
def step_action(d): _to_godot("step_action",     d)

# ── Scenario
@sio.on("load_scenario")
def load_scenario(d):  _to_godot("load_scenario", d)

@sio.on("reset_episode")
def reset_ep(d):       _to_godot("reset_episode", d)

# ── Diagnostics / overlays
@sio.on("set_planned_path")
def set_path(d):    _to_godot("set_planned_path", d)

@sio.on("set_intent_path")
def set_intent(d):  _to_godot("set_intent_path",  d)

@sio.on("set_state_label")
def set_label(d):   _to_godot("set_state_label",  d)

# ── Costmap / state polling
@sio.on("get_costmap")
def get_cmap(d):  _to_godot("get_costmap", d)

@sio.on("get_state")
def get_state(d): _to_godot("get_state",   d)


# ─────────────────────────────────────────────
#  HEALTH ENDPOINT
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return {
        "status": "ok",
        "uptime": _uptime(),
        "godot_connected": _godot_sid is not None,
        "clients": len(_client_sids)
    }


@app.route("/")
def index():
    return (
        "<h2>FOSSBot Simulation Server</h2>"
        f"<p>Uptime: {_uptime()}</p>"
        f"<p>Godot: {'✅ Connected' if _godot_sid else '❌ Not connected'}</p>"
        f"<p>Python clients: {len(_client_sids)}</p>"
        "<p><a href='/health'>/health</a> (JSON)</p>"
    )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FOSSBot SocketIO relay server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info(f"FOSSBot server starting on {args.host}:{args.port}")
    log.info("Godot:  connect with ?role=godot")
    log.info("Python: connect with ?role=client  (default)")

    sio.run(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
