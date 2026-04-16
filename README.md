# Anduril AI Grand Prix 🏁

**Duke University** | Team of 3 | Autonomous Drone Racing

A Python-based autonomy stack for the [Anduril AI Grand Prix](https://theaigrandprix.com) — a competition to fly a drone autonomously through a sequence of gates as fast as possible.

---

## Competition Overview

| Round | Timeline | Focus |
|---|---|---|
| Virtual Qualifier 1 | May 2026 | Completion — simplified course, clear gate visibility |
| Virtual Qualifier 2 | June 2026 | Speed — complex environment, no visual aids |
| Physical Qualifier | September 2026 | Real drones, California |
| Grand Prix Final | November 2026 | Real drones under race conditions, Ohio |

---

## Our Approach

We use a **progressive replacement** strategy — build a working pipeline in layers, then replace each layer with a smarter version.

```
Vision + Telemetry → Perception → Planning → Control → Pilot Commands
```

**Phase 1 (VQ1):** Classical CV for gate detection + PID control baseline  
**Phase 2 (VQ1):** Replace PID with a learned PPO policy  
**Phase 3 (VQ2):** Replace classical CV with CNN-based perception  

---

## Repository Structure

```
anduril-ai-grand-prix/
├── perception/              # Vision stack — gate detection & state estimation
│   ├── gate_detector.py     # Core detection module
│   ├── state_vector.py      # Normalized state vector output
│   └── tests/               # Unit tests for perception module
├── control/                 # Control layer
│   ├── pid_controller.py    # PID baseline
│   └── ppo_policy/          # Learned RL policy (Phase 2)
├── training/                # RL training scripts
│   ├── train_ppo.py
│   └── reward.py            # Reward function definitions
├── evaluation/              # Benchmarking & analysis
│   ├── metrics.py           # Lap time, gate pass rate, crash rate
│   └── compare_runs.ipynb   # W&B experiment comparison notebook
├── comms/                   # MAVLink interface
│   └── mavlink_bridge.py    # UDP SITL bridge (MAVLink v2 via MAVSDK)
├── docs/                    # Documentation
└── main.py                  # Entry point
```

---

## Technical Interface

The simulator communicates over **MAVLink v2 via UDP** (MAVSDK-compatible).

**Inputs from simulator:**
- Forward-facing FPV visual stream
- Telemetry: attitude, orientation, linear velocities (`ATTITUDE`, `HIGHRES_IMU`, `ODOMETRY`)

**Outputs to simulator:**
- `SET_POSITION_TARGET_LOCAL_NED` — position-based control
- `SET_ATTITUDE_TARGET` — attitude-based control (Throttle, Roll, Pitch, Yaw)

**Not available:** depth data, GPS/absolute position, engine RPMs, battery SoC

**Command rate:** 50–120 Hz | **Physics update rate:** 120 Hz

---

## Team

| Person | Role |
|---|---|
| Cameron | ML lead — RL policy, reward engineering, PID baseline |
| Parker | Perception — gate detection, state vector, CV pipeline |
| Pierson | Infrastructure — W&B experiment tracking, evaluation, analysis |

---

## State Vector Interface

The perception module outputs a normalized state vector consumed by the control layer:

```python
{
    "gate_center_x": float,   # Normalized -1.0 to 1.0 (left → right)
    "gate_center_y": float,   # Normalized -1.0 to 1.0 (top → bottom)
    "gate_size":     float,   # Proxy for distance — larger = closer
    "gate_angle":    float,   # Estimated rotation/tilt of gate
    "gate_visible":  bool,    # Whether a gate is currently detected
    "gate_distance": float,   # Estimated distance (calibrated from size)
}
```

---

## Setup

### Requirements

- Python 3.14.2+
- Windows machine with dedicated GPU (RTX 2060 Super or AMD RX 6600 XT minimum) for running the DCL simulator
- 16 GB RAM, 60 GB storage

### Installation

```bash
git clone https://github.com/<your-org>/anduril-ai-grand-prix.git
cd anduril-ai-grand-prix
pip install -r requirements.txt
```

### Dependencies

```
opencv-python
numpy
mavsdk
pymavlink
stable-baselines3
gymnasium
torch
wandb
```

### Running the Autonomy Stack

```bash
# Start the DCL simulator (Windows), then:
python main.py --mode pid          # Run PID baseline
python main.py --mode ppo          # Run learned RL policy
python main.py --mode eval         # Evaluation mode with metrics logging
```

### Training

```bash
# Train PPO policy on gym-pybullet-drones (pre-sim)
python training/train_ppo.py --env HoverAviary-v0

# Train on DCL sim (requires simulator running)
python training/train_ppo.py --env dcl
```

---

## Experiment Tracking

We use [Weights & Biases](https://wandb.ai) for experiment tracking. All training runs log:
- Reward over time
- Gate pass rate
- Lap time
- Crash rate
- Entropy & clip fraction (PPO diagnostics)

```bash
wandb login
# Runs are automatically logged during training
```

---

## Key Resources

**Papers**
- [Swift (Nature, 2023)](https://www.nature.com/articles/s41586-023-06419-4) — Champion-level drone racing with deep RL
- [Deep Drone Racing: Sim to Real (UZH, 2019)](https://rpg.ifi.uzh.ch/docs/RAL19_Kaufmann.pdf) — Reward design & sim-to-real transfer
- [MonoRace (MAVLab, 2025)](https://arxiv.org/abs/2503.16691) — End-to-end with minimal networks

**Code**
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — PPO reference implementation
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) — Pre-sim drone RL environment
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — RL baseline library
- [MAVSDK-Python](https://github.com/mavlink/MAVSDK-Python) — MAVLink interface

**Competition**
- [theaigrandprix.com](https://theaigrandprix.com)
- Technical Specification: VADR-TS-001 (Issue 00.01)

---

## Fair Play

Per competition rules, our submission:
- Contains no human interaction during timed runs
- Does not alter simulator game files
- Does not use screen tricks or exploit collision detection
- Is available for review if requested by organizers

---

## License

Code is owned by the team. By submitting to the AI Grand Prix, we grant the competition organizers permission to use our code strictly for operating, monitoring, and judging — no IP transfer to Anduril or any founding partner.
