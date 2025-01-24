import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.WCA2D import WCA2D
from cellular_automata.visualise import *

grid_shape = (10,10)
    
z = np.array([
    [0,0,0,2,2,2,2,0,0,0],
    [0,0,0,1.9,1.9,1.9,1.9,0,0,0],
    [0,0,0,1.8,1.8,1.8,1.8,1.8,0,0],
    [0,0,0,0,1.7,1.7,1.7,1.7,0,0],
    [0,0,0,0,1.6,1.6,1.6,1.6,0,0],
    [0,0,0,1.5,1.5,1.5,1.5,0,0,0],
    [0,0,0,1.4,1.4,1.4,1.4,0,0,0],
    [0,0,0,1.3,1.3,1.3,1.3,0,0,0],
    [0,0,0,1.2,1.2,1.2,1.2,0,0,0],
    [0,0,0,1.1,1.1,1.1,1.1,0,0,0]
])

d = np.array([
    [0,0,0,2,2,2,2,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0]
])
wall_bc = np.array([
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,1,0],
    [0,0,1,0,0,0,0,0,1,0],
    [0,0,1,1,0,0,0,0,1,0],
    [0,0,1,1,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,1,1,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,0,0]
])
vfr_in_bc = np.zeros(grid_shape)
vfr_in_bc[0,3:7] = 5
vfr_out_bc = np.zeros(grid_shape)
open_out_bc = np.zeros(grid_shape)
open_out_bc[-1,3:7] = True
porous_bc = np.zeros(grid_shape)

depth_tolerance = 0.01
n = 0.01

total_time = 100.0
dt = 0.1
max_dt = 0.2
output_interval = 0.5

wca = WCA2D(
    grid_shape, z, d, dx=1.0, depth_tolerance=depth_tolerance, n=n,
    wall_bc = wall_bc, vfr_in_bc=vfr_in_bc, vfr_out_bc=vfr_out_bc,
    open_out_bc=open_out_bc, porous_bc=porous_bc
)
wca.run_simulation(
    dt, max_dt, total_time=10.0, output_interval=output_interval, scheme="moore"
)

log = wca.log
ds = log.to_2D(log.d, grid_shape[0], grid_shape[1])
speed = log.speed()
dt = log.dt

window_size = 15
speed_avg = log.mv_avg(log.speed(),window_size)

plot_iteration_dependent_variable([dt],ylabels=["dt"])
visualize_cell_parameter(ds, zlabel='water depth', interval=100)
# visualize_cell_parameter_with_vector(
#     speed_avg,log.mv_avg(log.u(),window_size),log.mv_avg(log.v(-1),window_size),
#     zlabel="speed", interval=100, scale=0.03)