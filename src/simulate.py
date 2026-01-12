import mujoco 
import numpy as np
import mediapy as media

#Downloading model
model = mujoco.MjModel.from_xml_path("/Users/daeron/robust-mpc-mujoco/models/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)

#Model description 4 me 
nbody = model.nbody
nv = model.nv
nq = model.nq
print(f'Number of bodies: {nbody}')
print(f'Total number of DoFs in the model: {nv}')
print(f'Total number of generalized coordinates: {nq}')
print(f'Number of geoms in the scene: {model.ngeom}')
print(f'Colors of each geom: \n {model.geom_rgba}')

#simulation
data.qpos = np.array([0, 0, 0, 0, 0, 0])
duration = 5
framerate = 60

frames = []

renderer = mujoco.Renderer(model, width=640, height=480)

while data.time < duration:
    data.ctrl = np.array([-0.6,-1.2,-2.1,0.7,0.7,0.7])
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
media.write_video('simulation.mp4', frames, fps=framerate)
print("Видео сохранено в simulation.mp4")