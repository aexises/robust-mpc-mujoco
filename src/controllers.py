from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import scipy.linalg as la


@dataclass(frozen=True)
class LinearDynamics:
    A: np.ndarray
    B: np.ndarray

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def control_dim(self) -> int:
        return int(self.B.shape[1])


@dataclass(frozen=True)
class MPCSolution:
    first_action: np.ndarray
    action_plan: np.ndarray
    feasible: bool


def pd_controller(kp: float, kd: float, q: np.ndarray, q_des: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return kp * (q_des - q) - kd * qvel


def build_double_integrator_dynamics(num_joints: int, dt: float) -> LinearDynamics:
    identity = np.eye(num_joints)
    zeros = np.zeros((num_joints, num_joints))
    A = np.block([[identity, dt * identity], [zeros, identity]])
    B = np.block([[0.5 * dt**2 * identity], [dt * identity]])
    return LinearDynamics(A=A, B=B)


def build_prediction_matrices(dynamics: LinearDynamics, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    nx = dynamics.state_dim
    nu = dynamics.control_dim
    A_bar = np.zeros((nx * horizon, nx))
    B_bar = np.zeros((nx * horizon, nu * horizon))

    for step in range(horizon):
        row = slice(step * nx, (step + 1) * nx)
        A_bar[row] = np.linalg.matrix_power(dynamics.A, step + 1)

        for control_step in range(step + 1):
            col = slice(control_step * nu, (control_step + 1) * nu)
            B_bar[row, col] = np.linalg.matrix_power(dynamics.A, step - control_step) @ dynamics.B

    return A_bar, B_bar


def build_reference_trajectory(reference_state: np.ndarray, horizon: int) -> np.ndarray:
    return np.tile(reference_state, horizon)


def build_cost_matrices(
    num_joints: int,
    position_weight: float = 200.0,
    velocity_weight: float = 20.0,
    control_weight: float = 0.005,
    terminal_multiplier: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_cost = np.diag(
        np.concatenate(
            [
                position_weight * np.ones(num_joints),
                velocity_weight * np.ones(num_joints),
            ]
        )
    )
    control_cost = control_weight * np.eye(num_joints)
    terminal_cost = terminal_multiplier * state_cost
    return state_cost, control_cost, terminal_cost


def build_prediction_cost(
    state_cost: np.ndarray,
    control_cost: np.ndarray,
    terminal_cost: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    Q_bar = np.kron(np.eye(horizon), state_cost)
    Q_bar[-state_cost.shape[0] :, -state_cost.shape[1] :] = terminal_cost
    R_bar = np.kron(np.eye(horizon), control_cost)
    return Q_bar, R_bar


def compute_lqr_gain(dynamics: LinearDynamics, state_cost: np.ndarray, control_cost: np.ndarray) -> np.ndarray:
    P = la.solve_discrete_are(dynamics.A, dynamics.B, state_cost, control_cost)
    return -np.linalg.solve(control_cost + dynamics.B.T @ P @ dynamics.B, dynamics.B.T @ P @ dynamics.A)


def compute_rpi_bounds(
    closed_loop_matrix: np.ndarray,
    disturbance_bound: float,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> np.ndarray:
    spectral_radius = max(abs(np.linalg.eigvals(closed_loop_matrix)))
    if spectral_radius >= 1.0:
        raise ValueError("Closed-loop dynamics are not stable enough to compute tube bounds.")

    radius = np.zeros((closed_loop_matrix.shape[0],))
    power = np.eye(closed_loop_matrix.shape[0])

    for _ in range(max_iter):
        propagated = np.linalg.norm(power, ord=np.inf, axis=1) * disturbance_bound
        updated_radius = radius + propagated
        if np.all(np.abs(updated_radius - radius) < tol):
            return updated_radius
        radius = updated_radius
        power = power @ closed_loop_matrix

    return radius


def tighten_state_bounds(
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    tube_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    effective_tube = fit_tube_to_bounds(state_lower, state_upper, tube_bounds)
    tightened_lower = state_lower + effective_tube
    tightened_upper = state_upper - effective_tube
    return tightened_lower, tightened_upper


def fit_tube_to_bounds(
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    tube_bounds: np.ndarray,
    margin: float = 1e-6,
) -> np.ndarray:
    half_span = 0.5 * (state_upper - state_lower) - margin
    return np.minimum(tube_bounds, np.maximum(half_span, 0.0))


def solve_constrained_mpc(
    dynamics: LinearDynamics,
    horizon: int,
    x0: np.ndarray,
    x_ref: np.ndarray,
    state_cost: np.ndarray,
    control_cost: np.ndarray,
    terminal_cost: np.ndarray,
    state_lower: np.ndarray,
    state_upper: np.ndarray,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
) -> MPCSolution:
    A_bar, B_bar = build_prediction_matrices(dynamics, horizon)
    Q_bar, R_bar = build_prediction_cost(state_cost, control_cost, terminal_cost, horizon)
    H = B_bar.T @ Q_bar @ B_bar + R_bar
    H = 0.5 * (H + H.T)
    f = B_bar.T @ Q_bar @ (A_bar @ x0 - x_ref)

    U = cp.Variable(horizon * dynamics.control_dim)
    X_pred = A_bar @ x0 + B_bar @ U
    cost = 0.5 * cp.quad_form(U, H) + f.T @ U

    lower = np.tile(state_lower, horizon)
    upper = np.tile(state_upper, horizon)
    constraints = [
        X_pred >= lower,
        X_pred <= upper,
        U >= np.tile(control_lower, horizon),
        U <= np.tile(control_upper, horizon),
    ]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, warm_start=True)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or U.value is None:
        fallback = np.zeros(dynamics.control_dim)
        return MPCSolution(first_action=fallback, action_plan=np.zeros(horizon * dynamics.control_dim), feasible=False)

    action_plan = np.asarray(U.value).reshape(-1)
    first_action = action_plan[: dynamics.control_dim]
    return MPCSolution(first_action=first_action, action_plan=action_plan, feasible=True)
