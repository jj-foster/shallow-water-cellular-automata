import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.WCA2D import WCA2D
from cellular_automata.visualise import *

# Function to create a river bed with gentle slopes and an S curve
def create_river_bed(rows, cols, max_depth, slope_width, gradient, amp, bed_steepness):
    x = np.linspace(0, 2 * np.pi, rows)
    y_curve = np.sin(x) * (cols // (1/amp)) + cols // 2  # S-curve center line

    river_bed = np.zeros((rows, cols))
    z = np.zeros((rows, cols))
    for i in range(rows):
        center = int(y_curve[i])
        for j in range(cols):
            distance = abs(j - center)
            if distance < slope_width:
                # bed shape
                river_bed[i, j] = -max_depth * (1 - (distance / slope_width)**(2*bed_steepness))
                z[i, j] = river_bed[i, j]

            # downhill gradient
            z[i, j] += gradient * i

    return river_bed, z

grid_shape = (15,15)

max_depth = 3
river_width = grid_shape[0]/4
# gradient = -0.05
slope_grad = -0.1
S_amp = 0.15
bed_steepness = 2
depth, z = create_river_bed(
    grid_shape[0], grid_shape[1], max_depth, river_width, slope_grad, S_amp, bed_steepness
)

d = -depth * 0.55

# Plot the river bed
plt.imshow(-z, cmap='Blues', origin='upper', extent=(0, grid_shape[0], 0, grid_shape[1]))
plt.colorbar(label="Depth")
plt.title("River Bed with S Curve")
plt.xlabel("Width")
plt.ylabel("Length")
plt.show()

wall_bc = np.zeros(grid_shape)

vfr_in_bc = np.zeros(grid_shape)
vfr_in_bc[0,:] = -depth[0,:] * 5

vfr_out_bc = np.zeros(grid_shape)
# vfr_out_bc[-1,:] = vfr_in_bc[0,:]
vfr_out_bc[-1,:] = 50

open_out_bc = np.zeros(grid_shape)
porous_bc = np.zeros(grid_shape)

depth_tolerance = 0.01
n = 0.1

total_time = 2.0
dt = 0.01
max_dt = 1
output_interval = 0.1

wca = WCA2D(
    grid_shape, z, d, dx=1.0, depth_tolerance=depth_tolerance, n=n,
    wall_bc = wall_bc, vfr_in_bc=vfr_in_bc, vfr_out_bc=vfr_out_bc,
    open_out_bc=open_out_bc, porous_bc=porous_bc
)
wca.run_simulation(
    dt, max_dt, total_time=10.0, output_interval=output_interval, scheme="moore"
)

log = wca.log
speed = log.speed()
dt = log.dt

print(len(dt))

window_size = 5
speed_avg = log.mv_avg(log.speed(),window_size)
d_avg = log.mv_avg(log.d, window_size)
l_avg = log.mv_avg(log.l, window_size)
u_avg = log.mv_avg(log.u(), window_size)
v_avg = log.mv_avg(log.v(-1), window_size)

plot_iteration_dependent_variable([dt],ylabels=["dt"])
visualize_cell_parameter(d_avg, zlabel='water depth', interval=1)
visualize_cell_parameter_with_vector(speed_avg, u_avg, v_avg,
    zlabel="speed", interval=100, scale=0.03
)
