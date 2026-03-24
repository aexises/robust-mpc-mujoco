from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

try:
    from .actuators import ActuatorMotor, configure_all_actuators
    from .controllers import (
        build_cost_matrices,
        build_double_integrator_dynamics,
        build_reference_trajectory,
        compute_lqr_gain,
        compute_rpi_bounds,
        fit_tube_to_bounds,
        solve_constrained_mpc,
        tighten_state_bounds,
    )
except ImportError:
    from actuators import ActuatorMotor, configure_all_actuators
    from controllers import (
        build_cost_matrices,
        build_double_integrator_dynamics,
        build_reference_trajectory,
        compute_lqr_gain,
        compute_rpi_bounds,
        fit_tube_to_bounds,
        solve_constrained_mpc,
        tighten_state_bounds,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "universal_robots_ur5e" / "ur5e.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class SimulationConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    duration: float = 5.0
    horizon: int = 10
    framerate: int = 60
    render_video: bool = True
    save_plots: bool = True
    disturbance_bound: float = 1.0
    tube_disturbance_bound: float = 0.1
    velocity_limit: float = 10.0
    control_limit: float = 100.0
    position_weight: float = 200.0
    velocity_weight: float = 20.0
    control_weight: float = 0.005
    terminal_multiplier: float = 200.0
    seed: int | None = 7
    initial_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_positions: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, -0.8, 2.0, 2.0, 2.7, 10.7], dtype=float)
    )


@dataclass
class SimulationResult:
    time_steps: np.ndarray
    joint_positions: np.ndarray
    target_positions: np.ndarray
    control_inputs: np.ndarray
    infeasible_steps: np.ndarray
    tube_bounds: np.ndarray
    video_path: Path | None = None
    plot_paths: dict[str, Path] = field(default_factory=dict)


def _save_positions_plot(result: SimulationResult, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / "positions.png"
    plt.figure(figsize=(10, 6))
    for index in range(result.joint_positions.shape[1]):
        plt.plot(result.time_steps, result.joint_positions[:, index], label=f"Joint {index + 1}")
        plt.axhline(
            result.target_positions[index],
            color="r",
            linestyle="--",
            label="Target" if index == 0 else None,
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Joint Positions Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _save_error_plot(result: SimulationResult, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / "errors.png"
    errors = result.joint_positions - result.target_positions[None, :]
    position_tube = result.tube_bounds[: result.joint_positions.shape[1]]

    plt.figure(figsize=(10, 6))
    for index in range(result.joint_positions.shape[1]):
        plt.plot(result.time_steps, errors[:, index], label=f"Error Joint {index + 1}")
        plt.fill_between(
            result.time_steps,
            -position_tube[index],
            position_tube[index],
            color="g",
            alpha=0.15,
            label="Tube Bound" if index == 0 else None,
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (rad)")
    plt.title("Position Errors with Tube Bounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _save_controls_plot(result: SimulationResult, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / "controls.png"
    plt.figure(figsize=(10, 6))
    for index in range(result.control_inputs.shape[1]):
        plt.plot(result.time_steps, result.control_inputs[:, index], label=f"u Joint {index + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (acceleration-like)")
    plt.title("Control Inputs Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _save_infeasible_plot(result: SimulationResult, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / "infeasibility.png"
    plt.figure(figsize=(10, 3))
    plt.plot(result.time_steps, result.infeasible_steps.astype(int), "ro-", label="Infeasible")
    plt.xlabel("Time (s)")
    plt.ylabel("Flag")
    plt.title("QP Infeasibility Over Time")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_artifacts(result: SimulationResult, config: SimulationConfig, frames: list[np.ndarray]) -> SimulationResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.render_video and frames:
        import mediapy as media

        result.video_path = config.output_dir / "simulation.mp4"
        media.write_video(str(result.video_path), frames, fps=config.framerate)

    if config.save_plots:
        result.plot_paths = {
            "positions": _save_positions_plot(result, config.output_dir),
            "errors": _save_error_plot(result, config.output_dir),
            "controls": _save_controls_plot(result, config.output_dir),
            "infeasibility": _save_infeasible_plot(result, config.output_dir),
        }

    return result


def run_simulation(config: SimulationConfig) -> SimulationResult:
    model = mujoco.MjModel.from_xml_path(str(config.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    configure_all_actuators(model, ActuatorMotor(torque_range=(-config.control_limit, config.control_limit)))

    num_joints = model.nv
    dt = model.opt.timestep
    dynamics = build_double_integrator_dynamics(num_joints, dt)
    state_cost, control_cost, terminal_cost = build_cost_matrices(
        num_joints=num_joints,
        position_weight=config.position_weight,
        velocity_weight=config.velocity_weight,
        control_weight=config.control_weight,
        terminal_multiplier=config.terminal_multiplier,
    )
    closed_loop = dynamics.A + dynamics.B @ compute_lqr_gain(dynamics, state_cost, control_cost)
    tube_bounds = compute_rpi_bounds(closed_loop, disturbance_bound=config.tube_disturbance_bound)

    q_min = model.jnt_range[:, 0].copy()
    q_max = model.jnt_range[:, 1].copy()
    v_min = -config.velocity_limit * np.ones(num_joints)
    v_max = config.velocity_limit * np.ones(num_joints)
    x_min = np.concatenate([q_min, v_min])
    x_max = np.concatenate([q_max, v_max])
    effective_tube_bounds = fit_tube_to_bounds(x_min, x_max, tube_bounds)
    tightened_min, tightened_max = tighten_state_bounds(x_min, x_max, effective_tube_bounds)
    u_min = -config.control_limit * np.ones(num_joints)
    u_max = config.control_limit * np.ones(num_joints)

    data.qpos[:] = np.asarray(config.initial_positions, dtype=float)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=640, height=480) if config.render_video else None
    rng = np.random.default_rng(config.seed)
    frames: list[np.ndarray] = []
    joint_history: list[np.ndarray] = []
    control_history: list[np.ndarray] = []
    infeasible_history: list[bool] = []

    reference_state = np.concatenate([np.asarray(config.target_positions, dtype=float), np.zeros(num_joints)])
    stacked_reference = build_reference_trajectory(reference_state, config.horizon)

    while data.time < config.duration:
        q = data.qpos.copy()
        qvel = data.qvel.copy()
        x0 = np.concatenate([q, qvel])
        solution = solve_constrained_mpc(
            dynamics=dynamics,
            horizon=config.horizon,
            x0=x0,
            x_ref=stacked_reference,
            state_cost=state_cost,
            control_cost=control_cost,
            terminal_cost=terminal_cost,
            state_lower=tightened_min,
            state_upper=tightened_max,
            control_lower=u_min,
            control_upper=u_max,
        )

        mass_matrix = np.zeros((num_joints, num_joints))
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        bias_forces = data.qfrc_bias.copy()
        torque = mass_matrix @ solution.first_action + bias_forces
        disturbance = rng.uniform(-config.disturbance_bound, config.disturbance_bound, size=num_joints)

        data.ctrl[:] = torque
        data.qfrc_applied[:] = disturbance
        mujoco.mj_step(model, data)

        joint_history.append(data.qpos.copy())
        control_history.append(solution.first_action.copy())
        infeasible_history.append(not solution.feasible)

        if renderer is not None and len(frames) < data.time * config.framerate:
            renderer.update_scene(data)
            frames.append(renderer.render())

    if renderer is not None:
        renderer.close()

    joint_positions = np.asarray(joint_history)
    control_inputs = np.asarray(control_history)
    infeasible_steps = np.asarray(infeasible_history)
    time_steps = np.arange(1, len(joint_history) + 1) * dt

    result = SimulationResult(
        time_steps=time_steps,
        joint_positions=joint_positions,
        target_positions=np.asarray(config.target_positions, dtype=float),
        control_inputs=control_inputs,
        infeasible_steps=infeasible_steps,
        tube_bounds=effective_tube_bounds,
    )
    return save_artifacts(result, config, frames)
