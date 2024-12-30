import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cellular_automata.visualise import *

import numpy as np

class SWFCA_Model:
    def __init__(
            self, grid_shape, d, u, v, z, dx, CFL, manning_n,
            closed_bc = np.array([[0]]), inlet_bc = np.array([[0]]),
            flux_outlet_bc = np.array([[0]]), pressure_outlet_bc = np.array([[0]]),
            bh_tolerance=0.1, depth_threshold=0.1):
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
        assert grid_shape == d.shape, "Grid shape and water depth shape do not match"

        # Initialize fields
        self.d = d
        assert d.dtype == np.float64, "Water depth must be float64"
        self.u = u  # Velocity in x-direction
        assert u.dtype == np.float64, "u must be float64"
        self.v = v  # Velocity in y-direction
        assert v.dtype == np.float64, "v must be float64"
        self.z = z  # Bed elevation
        assert z.dtype == np.float64, "bed elevation must be float64"

        if np.any(closed_bc):
            self.closed_boundaries = closed_bc
        else:
            self.closed_boundaries = np.zeros(grid_shape, dtype=bool)
        if np.any(inlet_bc):
            self.inlet_boundaries = inlet_bc # inlet mass flux (Q, theta_idx)
        else:
            self.inlet_boundaries = np.zeros(grid_shape, dtype=bool)
        if np.any(flux_outlet_bc):
            self.flux_outlet_bc = flux_outlet_bc # outlet mass flux Q
        else:
            self.flux_outlet_bc = np.zeros(grid_shape, dtype=bool)
        if np.any(pressure_outlet_bc):
            self.pressure_outlet_bc = pressure_outlet_bc # outlet mass flux Q
        else:
            self.pressure_outlet_bc = np.zeros(grid_shape, dtype=bool)

        self.special_case = np.zeros(grid_shape, dtype=bool)

        self.n = manning_n  # Manning's roughness coefficient
        self.bh_tolerance = bh_tolerance
        self.depth_threshold = depth_threshold

        self.g = 9.81

        self.CFL = CFL  # Courant number
        self.dt = 0
        self.update_timestep()
        
        self.theta = (1, 1, -1, -1)
        self.direction_idx = [(0, 1), (-1, 0), (0, -1), (1, 0)]

        self.iteration = 0

    def update_timestep(self):
        min_dt = 100 # starting value for dt. represents max possible dt value
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.d[i, j] > self.depth_threshold:  # Only consider wet cells
                    local_dt = self.dx / (np.sqrt(self.u[i, j]**2 + self.v[i, j]**2) + np.sqrt(self.g * self.d[i, j]))
                    min_dt = min(min_dt, local_dt)

        self.dt = min_dt * self.CFL


    @staticmethod
    def compute_bernoulli_head(z, d, u, v):
        """Compute Bernoulli hydraulic head."""
        return z + d + 0.5 * (u**2 + v**2) / 9.81
    
    def is_greater_bh(self, H0, Hi):
        """Compare Bernoulli heads at neighbour cell."""
        return (H0 - Hi) >= self.bh_tolerance
    
    def is_outward_flow(self, Q, theta_idx):
        return (Q * self.theta[theta_idx]) >= 0
    
    def is_outward_flow_special_case(self, Q, theta_idx):
        return (Q * self.theta[theta_idx]) > 0
    
    def is_wet(self, d):
        """Check if the cell is dry."""
        return d >= self.depth_threshold

    def is_closed(self, i, j):
        """Check if the cell is closed."""
        return self.closed_boundaries[i, j]

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
        self.special_case = np.zeros_like(flow_dir, dtype=bool)

        # Iterate through cells and neighbors to calculate flow directions
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for theta_idx in range(len(self.theta)):

                    # Skip when the flow route points out of the grid
                    if (i == 0 and theta_idx == 1) \
                    or (j == 0 and theta_idx == 2) \
                    or (i == self.grid_shape[0]-1 and theta_idx == 3) \
                    or (j == self.grid_shape[1]-1 and theta_idx == 0):
                        flow_dir[i, j, theta_idx] = 0
                        continue

                    di, dj = self.direction_idx[theta_idx]
                    ni, nj = i + di, j + dj

                    if self.is_wet(d[i,j]) \
                    and self.is_greater_bh(bh[i, j], bh[ni, nj]) \
                    and self.is_outward_flow(flux[i, j, theta_idx], theta_idx) \
                    and not self.is_closed(i, j) \
                    and not self.is_closed(ni, nj): # apply closed boundary condition
                        flow_dir[i, j, theta_idx] = 1
                    else:
                        # special condition
                        if self.is_wet(d[i,j]) \
                        and self.is_wet(d[ni, nj]) \
                        and not self.is_greater_bh(bh[i, j], bh[ni, nj]) \
                        and self.is_outward_flow_special_case(flux[i, j, theta_idx], theta_idx) \
                        and not self.is_closed(i, j) \
                        and not self.is_closed(ni, nj): # apply closed boundary condition
                            flow_dir[i, j, theta_idx] = 1
                            self.special_case[i, j, theta_idx] = True
                            print("special case at", i, j, theta_idx,"at iteration", self.iteration)

        return flow_dir
    
    @staticmethod
    def flux_manning(n, l, d0, di, H0, Hi):
        """Calculate mass flux using Manning's equation.

        Args:
            d (float)): Centre cell depth
            d_n (float): Neighbour cell depth
            bh (float): Centre cell Bernoulli head
            bh_n (float): Neighbour cell Bernoulli head

        Returns:
            float: Manning's mass flux
        """
        edge_depth = (d0 + di) / 2

        return (1 / n) * (edge_depth**(-5/3)) * np.sqrt((H0 - Hi)/l)
    
    @staticmethod
    def flux_weir(l, H0, Hi, z0, zi):
        """Calculate mass flux using weir equation.

        Args:
            H0 (float): Centre cell Bernoulli head
            Hi (float): Neighbour cell Bernoulli head
            z0 (float): Centre cell bed elevation
            zi (float): Neighbour cell bed elevation

        Returns:
            float: Weir mass flux
        """

        max_z = max(z0, zi)
        h0 = H0 - max_z
        hi = max(0, Hi - max_z)
        psi = (1 - (hi/h0)**1.5)**0.385

        return 2/3 * l * np.sqrt(2 * 9.81) * psi * h0**1.5

    def special_case_flux(self, Q0i, H0, Hi, di):
        dQ0i = min(
            self.dx**2 / (2 * self.dt) * (Hi - H0 + self.bh_tolerance),
            self.dx**2 / self.dt * (di - self.depth_threshold)
        )

        new_Q0i = (Q0i / np.abs(Q0i)) * (np.abs(Q0i) - dQ0i)
        return new_Q0i

    def step2_update_mass_flux(self, flow_dir, bh, flux_prev):
        """Update the mass flux using Manning's and weir equations."""
        flux_new = np.zeros_like(flow_dir)

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for theta_idx in range(len(self.theta)):

                    di, dj = self.direction_idx[theta_idx]
                    ni, nj = i + di, j + dj

                    # flux depends on flow direction so no need to check for edge of grid as this was done in flow_dir
                    if flow_dir[i, j, theta_idx] == 1:
                        if not self.special_case[i, j, theta_idx]:
                            # normal flow case (H0>Hi, flux from centre to neighbor cell)
                            flux_manning = self.flux_manning(
                                self.n[i, j], self.dx, self.d[i, j], 
                                self.d[ni, nj], bh[i, j], bh[ni, nj]
                            )
                            flux_weir = self.flux_weir(
                                self.dx, bh[i, j], bh[ni, nj], self.z[i, j], self.z[ni, nj]
                            )

                            flux_new[i, j, theta_idx] = self.theta[theta_idx] * min(flux_manning, flux_weir)

                        else:   # special flow case (H0<Hi, flux from centre to neighbour cell)

                            flux_new[i, j, theta_idx] = self.special_case_flux(
                                flux_prev[i, j, theta_idx], bh[i, j], bh[ni, nj], self.d[ni, nj]
                            )

                            print(flux_prev[i,j,theta_idx], flux_new[i,j,theta_idx])

        return flux_new
    
    def flow_direction_unchanged(self, bh, zi, new_di, ui, vi):
        """Check if the flow direction remains unchanged."""
        return bh - self.bh_tolerance > zi + new_di + (ui**2 + vi**2) / (2*self.g)

    def water_depth_euler(self, flux, bh, flow_dir):
        """Predict water depth based on mass conservation."""
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            # new_d = np.copy(self.d)
            new_d = np.zeros_like(self.d)
            dd = np.zeros_like(self.d)
            negative_depth = False
            flow_dir_changed = False

            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    net_flux = 0
                    for idx, (di, dj) in enumerate(self.direction_idx):

                        # Skip when the flow route points out of the grid
                        if (i == 0 and idx == 1) \
                        or (j == 0 and idx == 2) \
                        or (i == self.grid_shape[0]-1 and idx == 3) \
                        or (j == self.grid_shape[1]-1 and idx == 0):
                            continue
                        
                        ni, nj = i + di, j + dj
                        # Adjust the sign based on the direction
                        if idx in [0, 1]:  # -Q01, -Q02
                            net_flux -= flux[i, j, idx]
                            # incoming flux from neighbour - should be 0 in this direction at current cell if flow_dir is correct
                            net_flux -= flux[ni, nj, (idx + 2) % 4]
                        else:  # +Q03, +Q04
                            net_flux += flux[i, j, idx]
                            net_flux += flux[ni, nj, (idx + 2) % 4]

                    dd[i, j] = self.dt * net_flux / (self.dx ** 2)
                    new_d[i, j] = self.d[i,j] + dd[i, j]

                    # Check for negative depth
                    if new_d[i, j] < 0:
                        negative_depth = True
                        break
                    elif new_d[i, j] < 0 and iteration == max_iterations - 1:
                        new_d[i, j] = 0
                        break
        
                if negative_depth:
                    break

            if negative_depth == False:
                # Check if flow direction remains unchanged
                for i in range(self.grid_shape[0]):
                    for j in range(self.grid_shape[1]):
                        for idx, (di, dj) in enumerate(self.direction_idx):
                            ni, nj = i + di, j + dj
                            if flow_dir[i, j, idx] == 1:
                                change_dir = self.flow_direction_unchanged(
                                    bh[i, j], self.z[ni, nj], new_d[ni, nj], self.u[ni, nj], self.v[ni, nj]
                                )
                                if (not change_dir and not self.special_case[i,j,idx]) \
                                or (change_dir and self.special_case[i,j,idx]):
                                    flow_dir_changed = True
                                    # print("AH",self.iteration,i,j)
                                    break
                        
                        if flow_dir_changed:
                            break
                    if flow_dir_changed:
                        break
                           
            if not negative_depth and not flow_dir_changed:
                break

            # Reduce the time step and recompute
            self.dt *= 0.5
            print(f"Reducing time step to {self.dt}; Iteration {iteration}")
            iteration += 1
        
        return new_d, dd

    @staticmethod
    def solve_quadratic(a, b, c):
        """Solve a quadratic equation."""
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 0, 0  # No real solution
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)
        return root1, root2

    def step4_predict_velocity(self, new_d, flow_dir, bh):
        """Predict water velocity based on the updated depth."""
    
        v_new = np.zeros((*self.grid_shape, 4))

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for idx, (di, dj) in enumerate(self.direction_idx):
                    ni, nj = i + di, j + dj

                    if flow_dir[i, j, idx] == 1:
                        # Compute the velocity
                        a = 1/(2 * self.g)
                        if idx == 0: # u(0,1)
                            b = (self.dx / 2) * ((self.n[ni, nj]**2 * np.abs(self.u[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.v[ni, nj]**2 / (2 * self.g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.u[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            root1, root2 = self.solve_quadratic(a, b, c)
                            if root1 > 0: v_new[i, j, 0] = root1
                            elif root2 > 0: v_new[i, j, 0] = root2
                            else : v_new[i, j, 0] = 0

                        elif idx == 1: # v(0,2)
                            b = (self.dx / 2) * ((self.n[ni, nj]**2 * np.abs(self.v[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.u[ni, nj]**2 / (2 * self.g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.v[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            root1, root2 = self.solve_quadratic(a, b, c)
                            if root1 > 0: v_new[i, j, 1] = root1
                            elif root2 > 0: v_new[i, j, 1] = root2
                            else : v_new[i, j, 1] = 0

                        elif idx == 2: # u(0,3)
                            b = (self.dx / 2) * ((self.n[ni, nj]**2 * np.abs(self.u[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.v[ni, nj]**2 / (2 * self.g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.u[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            root1, root2 = self.solve_quadratic(a, -b, c)
                            if root1 < 0: v_new[i, j, 2] = root1
                            elif root2 < 0: v_new[i, j, 2] = root2
                            else : v_new[i, j, 2] = 0

                        elif idx == 3: # v(0,4)
                            b = (self.dx / 2) * ((self.n[ni, nj]**2 * np.abs(self.v[ni, nj])) / (new_d[ni, nj]**(4/3)))
                            c = self.u[ni, nj]**2 / (2 * self.g) + new_d[ni, nj] + self.z[ni, nj] + (self.dx / 2) * (self.n[i, j]**2 * self.v[i, j]**2) / self.d[i, j]**(4/3) - bh[i, j]

                            root1, root2 = self.solve_quadratic(a, -b, c)
                            if root1 < 0: v_new[i, j, 3] = root1
                            elif root2 < 0: v_new[i, j, 3] = root2
                            else : v_new[i, j, 3] = 0
        
                        if v_new[i,j,idx]==0:
                            # print("0!",new_d[i,j],new_d[ni,nj])
                            print("0!", b, c, i,j,idx ,self.iteration)

        return v_new

    def step5_update_fields(self, d_new, v_new):
        """Update water depth, velocities, and Bernoulli hydraulic head."""
        self.d = d_new # update water depth
        
        # update velocities
        R = lambda u : 1 if u > 0 else 0
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if not self.is_closed(i, j):
                    u01 = v_new[i, j+1, 2] if j < self.grid_shape[1]-1 else 0
                    v02 = v_new[i-1, j, 3] if i > 0 else 0
                    u03 = v_new[i, j-1, 0] if j > 0 else 0                   
                    v04 = v_new[i+1, j, 1] if i < self.grid_shape[0]-1 else 0                   

                    self.u[i, j] = R(-u01) * u01 + R(u03) * u03
                    self.v[i, j] = R(-v02) * v02 + R(v04) * v04

        
    def inlet_boundary(self):
        """Applies inlet boundary condition. Assumes subcritical inlet flow
        """
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.inlet_boundaries[i, j, 0] != 0:
                    Q = self.inlet_boundaries[i, j, 0]
                    theta_idx = self.inlet_boundaries[i, j, 1]

                    # volume added to water depth
                    dV = Q * self.dt
                    self.d[i, j] += dV / self.dx ** 2

                    # update velocity
                    speed = Q / (self.dx * self.d[i, j])
                    if theta_idx == 0:
                        self.u[i, j] = speed
                    elif theta_idx == 1:
                        self.v[i, j] = speed
                    elif theta_idx == 2:
                        self.u[i, j] = -speed
                    elif theta_idx == 3:
                        self.v[i, j] = -speed
           
    def flux_outlet_boundary(self):
        """Applies outlet boundary condition. Assumes subcritical outlet flow
        """
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.flux_outlet_bc[i, j] != 0:
                    # Get outlet parameters
                    Q = self.flux_outlet_bc[i, j]  # Specified downstream depth

                    # Update water depth
                    dV = Q * self.dt
                    self.d[i, j] -= dV / self.dx**2
                    if self.d[i, j] < 0:
                        self.d[i, j] = 0

    def pressure_outlet_boundary(self, dd):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.pressure_outlet_bc[i, j] != 0:
                    # undo change in water depth
                    # (water passes in and straight ouf of cell)
                    self.d[i, j] -= dd[i, j]

                    # set 
                    self.u[i, j] = 0
                    self.v[i, j] = self.v[i-1, j]

    def closed_boundary(self):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.is_closed(i, j):
                    self.d[i,j] = 0
                    self.u[i,j] = 0
                    self.v[i,j] = 0

    def run_simulation(self, num_steps):
        """Run the simulation for a specified number of steps."""

        water_depths = [self.d.copy()]
        us = [self.u.copy()]
        vs = [self.v.copy()]
        dts = [self.dt]
        bhs = []

        flux = np.zeros((*self.grid_shape, 4))

        for step in range(num_steps):
            bh = self.compute_bernoulli_head(self.z, self.d, self.u, self.v)
            flow_dir = self.step1_determine_flow_direction(self.d, bh, flux)
            flux = self.step2_update_mass_flux(flow_dir, bh, flux_prev=flux)
            d_new, dd = self.water_depth_euler(flux, bh, flow_dir)
            # d_new = self.water_depth_rk4(flow_dir, bh)
            v_new = self.step4_predict_velocity(d_new, flow_dir, bh)
            self.step5_update_fields(d_new, v_new)
            
            if np.any(self.inlet_boundaries):
                self.inlet_boundary()
            if np.any(self.flux_outlet_bc):
                self.flux_outlet_boundary()
            if np.any(self.pressure_outlet_bc):
                self.pressure_outlet_boundary(dd)
            if np.any(self.closed_boundaries):
                self.closed_boundary()

            self.iteration += 1
            self.update_timestep()

            water_depths.append(self.d.copy())
            us.append(self.u.copy())
            vs.append(self.v.copy())
            dts.append(self.dt)
            bhs.append(bh)

        return water_depths, us, vs, dts, bhs