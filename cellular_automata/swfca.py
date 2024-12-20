# from .swfca import SWFCA_Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from numpy.typing import NDArray

# Constants and parameters
CFL = 0.9  # Courant number
g = 9.81  # Gravitational acceleration

class SWFCA_Model:
    def __init__(self, grid_shape, d, dx, dy, CFL, manning_n, bh_tolerance=0.1, depth_threshold=0.1):
        """_summary_

        Args:
            grid_shape (_type_): _description_
            dx (_type_): _description_
            dy (_type_): _description_
            dt (_type_): _description_
            manning_n (NDArray[np.float64]): 2d array of size (grid_shape[0], grid_shape[1])
            bh_tolerance (float, optional): _description_. Defaults to 0.1.
            depth_threshold (float, optional): _description_. Defaults to 0.1.
        """
        self.grid_shape = grid_shape
        self.dx = dx
        self.dy = dy
        assert dx == dy, "Only square cells are supported"

        # Initialize fields
        # self.d = np.zeros(grid_shape)  # Water depth
        self.d = d
        self.u = np.zeros(grid_shape)  # Velocity in x-direction
        self.v = np.zeros(grid_shape)  # Velocity in y-direction
        self.z = np.zeros(grid_shape)  # Bed elevation


        self.n = manning_n  # Manning's roughness coefficient
        self.bh_tolerance = bh_tolerance
        self.depth_threshold = depth_threshold

        self.CFL = CFL  # Courant number
        self.dt = 0
        self.update_timestep()

        self.theta = (1, 1, -1, -1)
        self.direction_idx = [(0, 1), (-1, 0), (0, -1), (1, 0)]


    def update_timestep(self):
        min_dt = float("inf")
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                if self.d[i, j] > self.depth_threshold:  # Only consider wet cells
                    local_dt = self.dx / (np.sqrt(self.u[i, j]**2 + self.v[i, j]**2) + np.sqrt(g * self.d[i, j]))
                    min_dt = min(min_dt, local_dt)

        self.dt = min_dt * self.CFL


    @staticmethod
    def compute_bernoulli_head(z, d, u, v):
        """Compute Bernoulli hydraulic head."""
        return z + d + 0.5 * (u**2 + v**2) / g
    
    def is_greater_bh(self, bh, bh_n):
        """Compare Bernoulli heads at neighbour cell."""
        return (bh - bh_n) >= self.bh_tolerance
    
    def is_outward_flow(self, q, theta_idx):

        return (q * self.theta[theta_idx]) >= 0
    
    def is_wet(self, d):
        """Check if the cell is dry."""
        return d >= self.depth_threshold

    def step1_determine_flow_direction(self, d, bh, flux):
        """_summary_

        Args:
            d (NDArray[np.float64]): Water depth. 2d array of size (grid_shape[0], grid_shape[1])
            bh (NDArray[np.float64]): Bernoulli head. 2d array of size (grid_shape[0], grid_shape[1])
            Q (NDArray[np.float64]): Mass flux. 3d array of size (grid_shape[0], grid_shape[1], 4)
            depth_threshold (float): Threshold depth for wet cells

        Returns:
            NDArray[np.float64]: Flow direction. 3d array of size (grid_shape[0], grid_shape[1], 4)
        """
        flow_dir = np.zeros_like(flux)

        # Iterate through cells and neighbors to calculate flow directions
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for theta_idx in range(len(self.theta)):
                    di, dj = self.direction_idx[theta_idx]
                    ni, nj = i + di, j + dj

                    if self.is_wet(d[i,j]) \
                    and self.is_wet(d[ni, nj]) \
                    and self.is_greater_bh(bh[i, j], bh[ni, nj]) \
                    and self.is_outward_flow(flux[i, j, theta_idx], theta_idx):
                        flow_dir[i, j, theta_idx] = 1

        return flow_dir
    
    @staticmethod
    def flux_manning(n, l, d, d_n, bh, bh_n):
        """Calculate mass flux using Manning's equation.

        Args:
            d (float)): Centre cell depth
            d_n (float): Neighbour cell depth
            bh (float): Centre cell Bernoulli head
            bh_n (float): Neighbour cell Bernoulli head

        Returns:
            float: Manning's mass flux
        """
        edge_depth = (d + d_n) / 2

        return (1 / n) * (edge_depth**(-5/3)) * np.sqrt((bh - bh_n)/l)
    
    @staticmethod
    def flux_weir(l, bh, bh_n, z, z_n):
        """Calculate mass flux using weir equation.

        Args:
            bh (float): Centre cell Bernoulli head
            bh_n (float): Neighbour cell Bernoulli head
            z (float): Centre cell bed elevation
            z_n (float): Neighbour cell bed elevation

        Returns:
            float: Weir mass flux
        """

        max_z = max(z, z_n)
        h0 = bh - max_z
        hi = max(0, bh_n - max_z)
        psi = (1 - (hi/h0)**1.5)**0.385

        return 2/3 * l * np.sqrt(2 * g) * psi * h0**1.5

    def step2_update_mass_flux(self, flow_dir, bh):
        """Update the mass flux using Manning's and weir equations."""
        flux = np.zeros_like(flow_dir)

        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for theta_idx in range(len(self.theta)):
                    flux_manning, flux_weir = 0, 0
                    if flow_dir[i, j, theta_idx] == 1:
                        di, dj = self.direction_idx[theta_idx]
                        ni, nj = i + di, j + dj

                        flux_manning = self.flux_manning(
                            self.n[i, j], self.dx, self.d[i, j], 
                            self.d[ni, nj], bh[i, j], bh[ni, nj]
                        )
                        flux_weir = self.flux_weir(
                            self.dx, bh[i, j], bh[ni, nj], self.z[i, j], self.z[ni, nj]
                        )

                    flux[i, j, theta_idx] = self.theta[theta_idx] * min(flux_manning, flux_weir)

        return flux
    
    def flow_direction_unchanged(self, bh, zi, di, ui, vi):
        """Check if the flow direction remains unchanged."""
        return bh - self.bh_tolerance >= zi + di + 0.5 * (ui**2 + vi**2) / g

    def step3_predict_water_depth(self, flux, bh, flow_dir):
        """Predict water depth based on mass conservation."""
        new_d = np.copy(self.d)
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            negative_depth = False
            flow_dir_changed = False

            for i in range(1, self.grid_shape[0] - 1):
                for j in range(1, self.grid_shape[1] - 1):
                    net_flux = 0
                    for idx, (di, dj) in enumerate(self.direction_idx):
                        ni, nj = i + di, j + dj
                        # Adjust the sign based on the direction
                        if idx in [0, 1]:  # -Q01, -Q02
                            net_flux -= flux[ni, nj, idx]
                        else:  # +Q03, +Q04
                            net_flux += flux[ni, nj, idx]

                    new_d[i, j] += self.dt * net_flux / (self.dx * self.dy)

                    # Check for negative depth
                    if new_d[i, j] < 0:
                        negative_depth = True
                        break

                    # Check if flow direction remains unchanged
                    for idx, (di, dj) in enumerate(self.direction_idx):
                        ni, nj = i + di, j + dj
                        if flow_dir[i, j, idx] == 1:
                            if not self.flow_direction_unchanged(
                                bh[i, j], self.z[ni, nj], self.d[ni, nj], self.u[ni, nj], self.v[ni, nj]
                            ):
                                print(bh[i, j] - self.bh_tolerance)
                                print(self.z[ni, nj] + self.d[ni, nj] + 0.5 * (self.u[ni, nj]**2 + self.v[ni, nj]**2) / g)
                                flow_dir_changed = True
                                break
                    
                if negative_depth or flow_dir_changed:
                    break
            
            if not negative_depth and not flow_dir_changed:
                break

            # Reduce the time step and recompute
            self.dt *= 0.5

            print(f"Reducing time step to {self.dt}; Iteration {iteration}")

        return new_d

    @staticmethod
    def solve_quadratic(a, b, c):
        """Solve a quadratic equation."""
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 0  # No real solution
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        return max(root1, root2, 0)  # Return the positive root or 0 if both are negative

    def step4_predict_velocity(self, new_d, flow_dir, bh):
        """Predict water velocity based on the updated depth."""
        u01, v02, u03, v04 = np.zeros_like(self.u), np.zeros_like(self.v), np.zeros_like(self.u), np.zeros_like(self.v)

        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for idx, (di, dj) in enumerate(self.direction_idx):
                    ni, nj = i + di, j + dj

                    if flow_dir[i, j, idx] == 1:
                        # Compute the velocity
                        a = 1/(2 * g)
                        if idx == 0: # u(0,1)
                            b = (1 / self.dx) * ((self.n[ni, nj]**2 * np.abs(self.u[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.v[ni, nj]**2 / (2 * g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.u[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            u01[i,j] = self.solve_quadratic(a, b, c)

                        elif idx == 1: # v(0,2)
                            b = (1 / self.dx) * ((self.n[ni, nj]**2 * np.abs(self.v[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.u[ni, nj]**2 / (2 * g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.v[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            v02[i,j] = self.solve_quadratic(a, b, c)

                        elif idx == 2: # u(0,3)
                            b = (1 / self.dx) * ((self.n[ni, nj]**2 * np.abs(self.u[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.v[ni, nj]**2 / (2 * g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.u[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            u03[i,j] = self.solve_quadratic(a, -b, c)

                        elif idx == 3: # v(0,4)
                            b = (1 / self.dx) * ((self.n[ni, nj]**2 * np.abs(self.v[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.u[ni, nj]**2 / (2 * g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.v[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            v04[i,j] = self.solve_quadratic(a, -b, c)

        return u01, v02, u03, v04

    def step5_update_fields(self, d_new, u01, v02, u03, v04):
        """Update water depth, velocities, and Bernoulli hydraulic head."""
        self.d = d_new # update water depth
        
        # update velocities
        R = lambda u : 1 if u > 0 else 0
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                self.u[i, j] = R(-u01[i,j]) * u01[i,j] + R(u03[i,j]) * u03[i,j]
                self.v[i, j] = R(-v02[i,j]) * v02[i,j] + R(v04[i,j]) * v04[i,j]

    def run_simulation(self, num_steps):
        """Run the simulation for a specified number of steps."""

        water_depths = []

        for step in range(num_steps):
            flux = np.zeros((*self.grid_shape, 4))
            bh = self.compute_bernoulli_head(self.z, self.d, self.u, self.v)
            flow_dir = self.step1_determine_flow_direction(self.d, bh, flux)
            flux = self.step2_update_mass_flux(flow_dir, bh)
            d_new = self.step3_predict_water_depth(flux, bh, flow_dir)
            u01, v02, u03, v04 = self.step4_predict_velocity(d_new, flow_dir, bh)
            self.step5_update_fields(d_new, u01, v02, u03, v04)

            self.update_timestep()
            print(self.dt)

            water_depths.append(self.d.copy())

        return water_depths

if __name__== "__main__":

    # Example usage
    grid_shape = (3, 4)
    dx, dy = 1.0, 1.0
    CFL = 0.5
    manning_n = np.full(grid_shape, 0.03)
    depth_threshold = 0.01

    num_steps = 10

    d = np.array([
        [0,0,0,0],
        [0,1,0.1,0],
        [0,0,0,0]
    ])

    model = SWFCA_Model(grid_shape, d, dx, dy, CFL, manning_n)
    water_depths = model.run_simulation(num_steps=num_steps)

    def print_grid(grid):
        for row in grid:
            print(" ".join(f"{cell:.2f}" for j, cell in enumerate(row)))
        print("\n")

    for i in range(num_steps):
        print_grid(water_depths[i])