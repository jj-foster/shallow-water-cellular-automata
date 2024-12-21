import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.swfca import SWFCA_Model
from cellular_automata.visualise import *

import numpy as np


grid_shape = (11,11)
dx = 1.0
CFL = 0.4
manning_n = np.full(grid_shape, 0.4)
depth_threshold = 0.01

num_steps = 100

d = np.full(grid_shape,1.0)
d[0,:] = 1.5
d[-1,:] = 1.5

z = np.zeros(grid_shape)

u = np.zeros(grid_shape)
v = np.zeros(grid_shape)
v[0,:] = -1.5
v[-1,:] = 1.5

closed_boundaries = np.zeros(grid_shape, dtype=bool)
# closed_boundaries[0,2] = True

inlet_bc = np.zeros(grid_shape + (2,))
# inlet_bc[0,0] = (0.1, 0)

model = SWFCA_Model(
    grid_shape, d, u, v, z, dx, CFL, manning_n,
    closed_bc=closed_boundaries, inlet_bc=inlet_bc
)
water_depths, us, vs, dt = model.run_simulation(num_steps=num_steps)

avg_water_depths = [np.mean(depth) for depth in water_depths]

visualize_cell_parameter(water_depths, interval=100)
# plot_iteration_dependent_variable(avg_water_depths)
plot_iteration_dependent_variable(dt,ylabel="dt (s)")
# visualize_water_depth_3d(water_depths, interval=100)