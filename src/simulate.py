import mujoco 
import numpy as np
import mediapy as media
#from controller import pd_controller
def pd_controller(kp, kd, q, qdes, qvel):
    return kp * (qdes - q) - kd * qvel

#Downloading model
model = mujoco.MjModel.from_xml_path("/Users/daeron/robust-mpc-mujoco/models/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data) 

#simulation
data.qpos = np.array([0, 0, 0, 0, 0, 0])
qdes = np.array([-1,-0.8,2,2,2.7,10.7])

duration = 5
framerate = 60

frames = []
history_q = []
renderer = mujoco.Renderer(model, width=640, height=480)

while data.time < duration:
    q = data.qpos.copy()
    qvel = data.qvel.copy()
    history_q.append(q)
    data.ctrl = pd_controller(10, 5, q, qdes, qvel)
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
media.write_video('simulation.mp4', frames, fps=framerate)
print("Видео сохранено в simulation.mp4")