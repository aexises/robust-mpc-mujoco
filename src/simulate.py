import mujoco 
import numpy as np
import mediapy as media

def pd_controller(kp, kd, q, qdes, qvel):
    return kp * (qdes - q) - kd * qvel

def predict_step(q, dq, v, dt):
    dq_next = dq + v * dt
    q_next  = q + dq * dt
    return q_next, dq_next

def stage_cost(q, dq, q_des, v):
    w_q  = 20.0   # точность
    w_dq = 1.0    # демпфирование
    w_v  = 0.1    # не дёргать приводы

    return (
        w_q  * np.sum((q - q_des)**2)
        + w_dq * np.sum(dq**2)
        + w_v  * np.sum(v**2)
    )

def rollout_cost(q0, dq0, v_seq, q_des, dt):
    q = q0.copy()
    dq = dq0.copy()
    cost = 0.0

    for v in v_seq:
        q, dq = predict_step(q, dq, v, dt)
        cost += stage_cost(q, dq, q_des, v)

    return cost

def mpc_step_1d(q, dq, q_des, dt):
    N = 1000
    v_candidates = [-5.0, 0.0, 5.0]

    best_cost = np.inf
    best_v0 = 0.0

    for v0 in v_candidates:
        for v1 in v_candidates:
            for v2 in v_candidates:

                v_seq = [v0, v1, v2]

                q_pred = q
                dq_pred = dq
                cost = 0.0

                for v in v_seq:
                    dq_pred = dq_pred + v * dt
                    q_pred = q_pred + dq_pred * dt
                    cost += (
                        20.0 * (q_pred - q_des)**2
                        + 1.0 * dq_pred**2
                        + 0.1 * v**2
                    )

                if cost < best_cost:
                    best_cost = cost
                    best_v0 = v0

    return best_v0

#Downloading model
model = mujoco.MjModel.from_xml_path("/Users/daeron/robust-mpc-mujoco/models/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)
nv = model.nv
mujoco.mj_resetData(model, data) 

#Actuators and motors

class ActuatorMotor:
    def __init__(self, torque_range = [-100,100]) -> None:
        self.range = torque_range
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])

    def __repr__(self) -> str:
        return f"ActuatorMotor(dyn={self.dyn}, gain={self.gain}, bias={self.bias})"


class ActuatorPosition(ActuatorMotor):
    def __init__(self, kp=1, kd=0, position_range = [-100,100]) -> None:
        super().__init__()
        self.range = position_range
        self.kp = kp
        self.kd = kd
        self.gain[0] = self.kp
        self.bias[1] = -self.kp
        self.bias[2] = -self.kd


class ActuatorVelocity(ActuatorMotor):
    def __init__(self, kv=1,  velocity_range = [-100,100]) -> None:
        super().__init__()
        self.range = velocity_range
        self.kv = kv
        self.gain[0] = self.kv
        self.bias[2] = -self.kv

def update_actuator(model, actuator_id, actuator):
    """
    Update actuator in model
    model - mujoco.MjModel
    actuator_id - int or str (name) (for reference see, named access to model elements)
    actuator - ActuatorMotor, ActuatorPosition, ActuatorVelocity
    """
    model.actuator(actuator_id).dynprm = np.zeros(len(model.actuator(actuator_id).dynprm))
    model.actuator(actuator_id).gainprm = np.zeros(len(model.actuator(actuator_id).gainprm))
    model.actuator(actuator_id).biasprm = np.zeros(len(model.actuator(actuator_id).biasprm))
    model.actuator(actuator_id).ctrlrange = actuator.range 
    model.actuator(actuator_id).dynprm[:3] = actuator.dyn
    model.actuator(actuator_id).gainprm[:3] = actuator.gain
    model.actuator(actuator_id).biasprm[:3] = actuator.bias


# update actuators
position_motor = ActuatorMotor()

for actuator_id in range(model.nu):
    update_actuator(model, actuator_id, position_motor)

#Simulation
data.qpos = np.array([0, 0, 0, 0, 0, 0])
qdes = np.array([-1,-0.8,2,2,2.7,10.7])

duration = 5
framerate = 60

frames = []
history_q = []
renderer = mujoco.Renderer(model, width=640, height=480)
mpc_joint = 0
dt = model.opt.timestep

while data.time < duration:
    q = data.qpos.copy()
    qvel = data.qvel.copy()
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    h = data.qfrc_bias.copy()
    #now applying MPC
    v = np.zeros(nv)
    v_mpc = mpc_step_1d(q[mpc_joint], qvel[mpc_joint], qdes[mpc_joint], dt)
    v[mpc_joint] = v_mpc
    tau = M @ v + h
    data.ctrl = tau
    mujoco.mj_step(model, data)
    history_q.append(q)
    if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
media.write_video('simulation.mp4', frames, fps=framerate)
print("Video saved as simulation.mp4")