import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.swfca import SWFCA_Model
from cellular_automata.visualise import *

import numpy as np


grid_shape = (7,7)
dx = 1.0
CFL = 0.1
manning_n = np.full(grid_shape, 0.0003)
depth_threshold = 0.01
bh_tolerance = 0.01

num_steps = 4

d = np.full(grid_shape,1.0)
d[0,0] = 1.3

z = np.zeros(grid_shape)

u = np.zeros(grid_shape)
v = np.zeros(grid_shape)

closed_boundaries = np.zeros(grid_shape, dtype=bool)
# closed_boundaries[0,2] = True

inlet_bc = np.zeros(grid_shape + (2,))
# inlet_bc[0,0] = (0.1, 0)

model = SWFCA_Model(
    grid_shape, d, u, v, z, dx, CFL, manning_n,
    closed_bc=closed_boundaries, inlet_bc=inlet_bc, 
    depth_threshold=depth_threshold, bh_tolerance=bh_tolerance
)
ds, us, vs, dt, bh = model.run_simulation(num_steps=num_steps)
avg_water_depths = [np.mean(depth) for depth in ds]
std_bh = [np.std(h) for h in bh]
print(std_bh[-1])

speed = [np.sqrt(us[t]**2 + vs[t]**2) for t in range(num_steps)]

# plot_iteration_dependent_variable([dt, std_bh], ["dt (s)", "Hydrodynamic head std"])

visualize_cell_parameter(bh, interval=1000)
# visualize_cell_parameter(ds, interval=1000)
# visualize_cell_parameter(speed, interval=1000)
# visualize_water_depth_3d(water_depths, interval=100)