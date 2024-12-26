import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.swfca import SWFCA_Model
from cellular_automata.visualise import *

import numpy as np


grid_shape = (5,1)
dx = 1.0
CFL = 0.3
manning_n = np.full(grid_shape, 0.01)
depth_threshold = 0.01
bh_tolerance = 0.01

num_steps = 100

d = np.full(grid_shape,0.1)
d[0,0]=0.2

z = np.zeros(grid_shape)
z = np.array([
    [0.5], [0.4], [0.3], [0.2], [0.1]
])
u = np.zeros(grid_shape)
v = np.zeros(grid_shape)

closed_boundaries = np.zeros(grid_shape, dtype=bool)

inlet_bc = np.zeros(grid_shape + (2,))
# inlet_bc[0,0] = (0.1, 3)

pressure_outlet_bc = np.zeros(grid_shape)
pressure_outlet_bc[-1,0] = True

model = SWFCA_Model(
    grid_shape, d, u, v, z, dx, CFL, manning_n,
    closed_bc=closed_boundaries, inlet_bc=inlet_bc,
    pressure_outlet_bc=pressure_outlet_bc,
    depth_threshold=depth_threshold, bh_tolerance=bh_tolerance
)
ds, us, vs, dt, bhs = model.run_simulation(num_steps=num_steps)

avg_water_depths = [np.mean(depth) for depth in ds]
std_bh = [np.std(h) for h in bhs]
print(std_bh[-1])

plot_iteration_dependent_variable([dt, std_bh], ["dt (s)", "Hydrodynamic head std"])

# visualize_cell_parameter(bhs, interval=100)
# visualize_cell_parameter(ds, interval=100)
visualize_cell_parameter(vs, interval=100)
# visualize_water_depth_3d(ds, interval=100)

v_avg = -np.mean([v[-20:,0] for v in vs],axis=0)
d_avg = np.mean([d[-20:,0] for d in ds], axis=0)
import matplotlib.pyplot as plt
plt.plot(v_avg)
plt.plot(d_avg)
plt.show()