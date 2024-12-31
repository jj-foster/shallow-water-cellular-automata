import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.swfca import SWFCA_Model
from cellular_automata.visualise import *

import numpy as np


grid_shape = (10,10)
dx = 1.0
CFL = 0.5
manning_n = np.full(grid_shape, 0.1)
vel_n = np.full(grid_shape, 2)
depth_threshold = 0.01
bh_tolerance = 0.01

num_steps = 100

d = np.full(grid_shape,0.0)
d = np.array([
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0]
],dtype=np.float64)

z = np.zeros(grid_shape)

u = np.zeros(grid_shape)
v = np.zeros(grid_shape)
# v[:,3:7] = -0.1
v = np.full(grid_shape, -0.1, dtype=np.float64)

closed_boundaries = np.zeros(grid_shape, dtype=bool)
closed_boundaries = np.array([
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,1,0],
    [0,0,0,1,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0]
])

inlet_bc = np.zeros(grid_shape + (2,))
inlet_bc[0,3:7] = (1, 3)

pressure_outlet_bc = np.zeros(grid_shape)
pressure_outlet_bc[-1,3:7] = True

model = SWFCA_Model(
    grid_shape, d, u, v, z, dx, CFL, manning_n,
    closed_bc=closed_boundaries, inlet_bc=inlet_bc,
    pressure_outlet_bc=pressure_outlet_bc,
    depth_threshold=depth_threshold, bh_tolerance=bh_tolerance, vel_n=vel_n
)
ds, us, vs, dt, bhs = model.run_simulation(num_steps=num_steps)

avg_water_depths = [np.mean(depth) for depth in ds]
std_bh = [np.std(h) for h in bhs]
print(std_bh[-1])

speed = [np.sqrt(us[t]**2 + vs[t]**2) for t in range(len(ds))]
plot_iteration_dependent_variable([dt, std_bh], ["dt (s)", "Hydrodynamic head std"])

visualize_cell_parameter(bhs, interval=100)
visualize_cell_parameter(ds, interval=100)
visualize_cell_parameter(speed, interval=100)
# visualize_water_depth_3d(ds, interval=100)

v_avg = -np.mean([v[-20:,3] for v in vs],axis=0)
d_avg = np.mean([d[-20:,3] for d in ds], axis=0)
# print(v_avg)
import matplotlib.pyplot as plt
plt.plot(v_avg, label="v_avg")
plt.plot(d_avg, label="d_avg")
plt.legend()
# plt.show()