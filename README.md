# Robust MPC for Robotic Manipulation in MuJoCo

This project studies model-based MPC for robotic manipulators under model uncertainty and disturbances.

## Repository layout

- `src/actuators.py`: actuator configuration helpers for MuJoCo models.
- `src/controllers.py`: linear dynamics helpers, LQR utilities, and constrained MPC solving.
- `src/simulation.py`: simulation orchestration, artifact generation, and result objects.
- `src/simulate.py`: small CLI entry point for running experiments.
- `models/`: MuJoCo robot descriptions.
- `artifacts/`: generated videos and plots created by the simulation runner.

## Run

```bash
python3 src/simulate.py
```

Useful options:

```bash
python3 src/simulate.py --duration 1.0 --no-render
python3 src/simulate.py --output-dir artifacts/debug --horizon 15
```
