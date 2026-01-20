import mujoco 
import numpy as np
import mediapy as media
import scipy.linalg as la
import cvxpy as cp

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

def compute_lqr_gain(A, B, Q, R):
    P = la.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K 

K = compute_lqr_gain(A, B, Q, R)
A_cl = A + B @ K

def compute_rpi_bounds(A_cl, W_max, tol = 1e-4, max_iter = 1000):
    rho = max(abs(np.linalg.eigvals(A_cl)))
    if rho > 1:
        raise ValueError("A_cl is not stable")
    
    # Iterative sum for box (using inf-norm)
    Omega = np.zeros((2*n,))  # Start with zero set
    Ak = np.eye(2*n)  # A_cl^0
    for k in range(max_iter):
        add_set = np.linalg.norm(Ak, ord=np.inf, axis=1) * W_max  # Propagate W bound
        new_Omega = Omega + add_set
        if np.all(np.abs(new_Omega - Omega) < tol):
            break
        Omega = new_Omega
        Ak = Ak @ A_cl
    bounds = Omega  # Symmetric, so [-bounds, bounds]
    return bounds  # Shape (12,), radius per state

def solve_constrained_mpc(A_bar, B_bar, H, f, x0, x_ref, x_min_tight, x_max_tight, u_min, u_max, N, n, nx):
    nu = n  # inputs per step
    U = cp.Variable(N * nu)  # optimization var
    cost = 0.5 * cp.quad_form(U, H) + f.T @ U
    X_pred = A_bar @ x0 + B_bar @ U
    
    constraints = [
        X_pred >= x_min_tight,  # State lower
        X_pred <= x_max_tight,  # State upper
        U >= np.tile(u_min, N), 
        U <= np.tile(u_max, N)
    ]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)  # Or ECOS if issues
    
    if prob.status != cp.OPTIMAL:
        print("QP infeasible!")  # Fallback: use unconstrained or zero u
        return np.zeros(N * nu)  # Or last U
    
    return U.value

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

# Position bounds 
q_min = model.jnt_range[:, 0].copy()  # Lower joint limits
q_max = model.jnt_range[:, 1].copy()  # Upper

# Velocity bounds 
v_min = -10.0 * np.ones(n)  # rad/s
v_max = 10.0 * np.ones(n)

# State bounds
x_min = np.concatenate([q_min, v_min])
x_max = np.concatenate([q_max, v_max])

# Input bounds (u as accel)
u_min = -100.0 * np.ones(n)
u_max = 100.0 * np.ones(n)

# Tube bounds
tube_bounds = compute_rpi_bounds(A_cl, W_max = 0.1)

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
    N = 10
    x0 = np.concatenate([q, qvel])
    x_ref_single = np.concatenate([qdes, np.zeros(n)])
    x_ref = np.tile(x_ref_single, N)
    A_bar, B_bar = build_prediction_matrices(A, B, N)
    Q_bar = np.kron(np.eye(N), Q)
    Q_bar[-Q.shape[0]:, -Q.shape[1]:] = Qf
    R_bar = np.kron(np.eye(N), R)
    H = B_bar.T @ Q_bar @ B_bar + R_bar
    f = B_bar.T @ Q_bar @ (A_bar @ x0 - x_ref)
    x_min_tight = np.tile(x_min + tube_bounds, N)
    x_max_tight = np.tile(x_max - tube_bounds, N)
    U = solve_constrained_mpc(A_bar, B_bar, H, f, x0, x_ref, x_min_tight, x_max_tight, u_min, u_max, N, n, 2*n)
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
