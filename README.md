# FOSSBot Simulation Platform - PoC

A deterministic robotics simulation platform with step-synchronized
control, modular sensors, and reinforcement learning integration.

## What Works

**Step-Sync Core**

* Deterministic stepping with state aggregation
* Run: See `demo_navigation.py`

**Modular Sensors**

* Independent sensor nodes (sensors/ folder)
* Easy to add new sensors

**YAML Scenarios**

* Data-driven environment configuration
* Example: scenarios/

**Gymnasium Wrapper**

* RL-compatible interface
* Works with Stable-Baselines3

**Safety Layer**

* Collision detection, watchdog timer
* Prevents unsafe robot behavior

 

## Quick Start

```bash
# Setup
pip install -r requirements.txt
# Edit path to Godot instance in fossbot_client.py

# Run demo
python python/demo_navigation.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 
