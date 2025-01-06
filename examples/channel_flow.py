import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.WCA2D import WCA2D
from cellular_automata.visualise import *

grid_shape = (2, 5)
    
z = np.zeros(grid_shape)

d = np.full(grid_shape, 1.0)

wall_bc = np.zeros(grid_shape)
vfr_in_bc = np.zeros(grid_shape)
vfr_in_bc[:,0] = 1
vfr_out_bc = np.zeros(grid_shape)
open_out_bc = np.zeros(grid_shape)
open_out_bc[:,-1] = True
porous_bc = np.zeros(grid_shape)

depth_tolerance = 0.01
n = 0.1

total_time = 10.0
dt = 0.1
max_dt = 0.5
output_interval = 0.1

wca = WCA2D(
    grid_shape, z, d, dx=1.0, depth_tolerance=depth_tolerance, n=n,
    wall_bc = wall_bc, vfr_in_bc=vfr_in_bc, vfr_out_bc=vfr_out_bc,
    open_out_bc=open_out_bc, porous_bc=porous_bc
)
wca.run_simulation(
    dt, max_dt, total_time=10.0, output_interval=output_interval, scheme="von_neumann"
)

log = wca.log
ds = log.d
speed = log.speed()

visualize_cell_parameter(ds, zlabel='water depth', interval=100)
visualize_cell_parameter(speed, zlabel='speed', interval=100)