from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActuatorMotor:
    torque_range: tuple[float, float] = (-100.0, 100.0)
    dyn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    gain: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    bias: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    @property
    def ctrl_range(self) -> np.ndarray:
        return np.asarray(self.torque_range, dtype=float)


@dataclass
class ActuatorPosition(ActuatorMotor):
    kp: float = 1.0
    kd: float = 0.0
    torque_range: tuple[float, float] = (-100.0, 100.0)

    def __post_init__(self) -> None:
        self.gain = np.array([self.kp, 0.0, 0.0])
        self.bias = np.array([0.0, -self.kp, -self.kd])


@dataclass
class ActuatorVelocity(ActuatorMotor):
    kv: float = 1.0
    torque_range: tuple[float, float] = (-100.0, 100.0)

    def __post_init__(self) -> None:
        self.gain = np.array([self.kv, 0.0, 0.0])
        self.bias = np.array([0.0, 0.0, -self.kv])


def update_actuator(model, actuator_id: int, actuator: ActuatorMotor) -> None:
    target = model.actuator(actuator_id)
    target.dynprm = np.zeros_like(target.dynprm)
    target.gainprm = np.zeros_like(target.gainprm)
    target.biasprm = np.zeros_like(target.biasprm)
    target.ctrlrange = actuator.ctrl_range
    target.dynprm[:3] = actuator.dyn
    target.gainprm[:3] = actuator.gain
    target.biasprm[:3] = actuator.bias


def configure_all_actuators(model, actuator: ActuatorMotor | None = None) -> None:
    selected_actuator = actuator or ActuatorMotor()
    for actuator_id in range(model.nu):
        update_actuator(model, actuator_id, selected_actuator)
