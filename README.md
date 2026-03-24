# Robust MPC for Robotic Manipulation in MuJoCo

This project studies model-based MPC for robotic manipulators under model uncertainty and disturbances.

## Main workspace

The main way to work with this project is the notebook:

- `notebooks/robust_mpc_workbench.ipynb`: choose parameters, run simulations, generate video, inspect plots, and keep experiment notes.

The scripts under `src/` are the reusable inner engine used by the notebook.

## Repository layout

- `src/actuators.py`: actuator configuration helpers for MuJoCo models.
- `src/controllers.py`: linear dynamics helpers, LQR utilities, and constrained MPC solving.
- `src/simulation.py`: simulation orchestration, artifact generation, and result objects.
- `src/simulate.py`: thin CLI entry point for running experiments outside the notebook.
- `models/`: MuJoCo robot descriptions.
- `artifacts/`: generated videos and plots created by the simulation runner.

## Run from notebook

```bash
jupyter notebook notebooks/robust_mpc_workbench.ipynb
```

## Run from CLI

```bash
python3 src/simulate.py
python3 src/simulate.py --duration 1.0 --no-render
python3 src/simulate.py --output-dir artifacts/debug --horizon 15
```
