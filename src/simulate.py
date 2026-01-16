import mujoco 
import numpy as np
import mediapy as media

#Downloading model
model = mujoco.MjModel.from_xml_path("/Users/daeron/robust-mpc-mujoco/models/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)
nv = model.nv
mujoco.mj_resetData(model, data) 



dt = model.opt.timestep
n = model.nv  # 6

A = np.block([
    [np.eye(n), dt * np.eye(n)],
    [np.zeros((n, n)), np.eye(n)]
])

B = np.block([
    [0.5 * dt**2 * np.eye(n)],
    [dt * np.eye(n)]
])

def build_prediction_matrices(A, B, N):
    nx = A.shape[0]
    nu = B.shape[1]

    A_bar = np.zeros((nx*N, nx))
    B_bar = np.zeros((nx*N, nu*N))

    for i in range(N):
        A_bar[i*nx:(i+1)*nx] = np.linalg.matrix_power(A, i+1)

        for j in range(i+1):
            B_bar[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = \
                np.linalg.matrix_power(A, i-j) @ B

    return A_bar, B_bar

Q = np.diag(
    np.concatenate([
        50.0 * np.ones(n),   # позиция
        5.0  * np.ones(n)    # скорость
    ])
)

R = 0.01 * np.eye(n)
Qf = 200.0 * Q



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
    N = 20
    x0 = np.concatenate([q, qvel])
    x_ref_single = np.concatenate([qdes, np.zeros(n)])
    x_ref = np.tile(x_ref_single, N)
    A_bar, B_bar = build_prediction_matrices(A, B, N)
    Q_bar = np.kron(np.eye(N), Q)
    Q_bar[-Q.shape[0]:, -Q.shape[1]:] = Qf
    R_bar = np.kron(np.eye(N), R)
    H = B_bar.T @ Q_bar @ B_bar + R_bar
    f = B_bar.T @ Q_bar @ (A_bar @ x0 - x_ref)
    U = -np.linalg.solve(H, f)
    u0 = U[:n]
    tau = M @ u0 + h
    data.ctrl = tau
    #adding disturbance
    disturbance = np.random.uniform(-1.0, 1.0, size=nv)
    data.qfrc_applied[:] += disturbance
    mujoco.mj_step(model, data)
    history_q.append(q)
    if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
media.write_video('simulationDIST.mp4', frames, fps=framerate)
print("Video saved as simulation.mp4")
