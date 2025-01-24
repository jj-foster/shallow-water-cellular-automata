import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.WCA2D import WCA2D
from cellular_automata.visualise import *

grid_shape = (5, 5)
    
z = np.zeros(grid_shape)

d = np.full(grid_shape, 0.0)
# d[0, 0] = 1.0

wall_bc = np.zeros(grid_shape)
vfr_in_bc = np.zeros(grid_shape)
vfr_in_bc[0, 0] = 5.0
vfr_out_bc = np.zeros(grid_shape)
vfr_out_bc[-1, -1] = 5.0
open_out_bc = np.zeros(grid_shape)
porous_bc = np.zeros(grid_shape)

depth_tolerance = 0.01
n = 0.5

total_time = 20.0
dt = 0.1
max_dt = 1.0
output_interval = 0.2

wca = WCA2D(
    grid_shape, z, d, dx=1.0, depth_tolerance=depth_tolerance, n=n,
    wall_bc = wall_bc, vfr_in_bc=vfr_in_bc, vfr_out_bc=vfr_out_bc,
    open_out_bc=open_out_bc, porous_bc=porous_bc
)
wca.run_simulation(
    dt, max_dt, total_time=total_time, output_interval=output_interval,
    scheme="moore"
)

log = wca.log
ds = log.to_2D(log.d, grid_shape[0], grid_shape[1])

visualize_cell_parameter(ds, zlabel='water depth', interval=100)