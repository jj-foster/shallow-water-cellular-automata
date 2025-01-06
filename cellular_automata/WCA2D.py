import numpy as np

class WCA2D:
    def __init__(
            self, grid_shape, z, d, dx, depth_tolerance, n,
            wall_bc, vfr_in_bc, vfr_out_bc, open_out_bc, porous_bc
        ):
        """_summary_

        Args:
            grid_shape (tuple): #rows, #cols
            z (np.ndarray): Bed elevation. Shape like grid_shape
            d (_type_): Initial water depths. Shape like grid_shape
            dx (float): Rectangular cell width (m)
            depth_tolerance (float): Minimum water depth for downstream route to be considered
            n (float): Manning friction factor
            wall_bc (np.ndarray): Zero-flux boundary condition. Shape like grid_shape. True/False per cell
            vfr_in_bc (np.ndarray): Volumetric flow rate inflow boundary condition. Shape like grid_shape. dV/dt (m3/s)
            vfr_out_bc (np.ndarray): Volumetric flow rate outflow boundary condition. Shape like grid_shape. dV/dt (m3/s)
            open_out_bc (np.ndarray): Open outlet boundary condition. Shape like grid_shape. True/False per cell.
            porous_bc (np.ndarray): Limits intercellular volume output from cell. Shape like grid_shape. 0 to 1 proportion of flow blocked. 1 = no flow through cell 
        """
        self.grid_shape = grid_shape
        self.z = z
        self.d = d
        self.l = z + d
        self.dx = dx
        self.depth_tolerance = depth_tolerance
        self.n = n
        self.I_total = np.zeros_like(self.d)

        self.wall_bc = wall_bc
        self.vfr_in_bc = vfr_in_bc
        self.vfr_out_bc = vfr_out_bc
        self.open_out_bc = open_out_bc
        self.porous_bc = porous_bc
    
    def get_neighbours(self, row, col, scheme):
        """
        Get neighbors for a given cell in the grid based on the chosen scheme.
        
        Parameters:
            row (int): Row index of the central cell.
            col (int): Column index of the central cell.
            scheme (str): The neighborhood scheme to use ('von_neumann' or 'moore').
            
        Returns:
            List of tuples (neighbor_row, neighbor_col, direction_idx) representing neighbor indices.
        """
        if scheme == "von_neumann":
            # von Neumann neighbors (North, South, West, East)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif scheme == "moore":
            # Moore neighbors (All surrounding cells)
            directions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),         (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        else:
            raise ValueError(f"Unsupported neighborhood scheme: {scheme}")
        
        neighbors = []
        for direction_idx, (dr, dc) in enumerate(directions):
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                if not self.wall_bc[r, c]:
                    neighbors.append((dr, dc, direction_idx))
        
        return neighbors
    
    def compute_intercellular_volume(self, scheme):
        """
        Compute intercellular volumes for all cells in the grid.
        Updates water depth for the next time step.
        """
        new_d = np.copy(self.d)  # To store updated water depths
        new_I_total = np.zeros_like(self.I_total)  # To store updated total intercellular volumes

        max_neighbours = 8 if scheme == "moore" else 4
        I_ij = np.zeros((self.grid_shape[0], self.grid_shape[1], max_neighbours), dtype=np.float64)

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):

                if self.wall_bc[row, col]:
                    continue

                # Get central cell properties
                l0 = self.l[row, col]
                d0 = self.d[row, col]

                # Identify downstream neighbors
                neighbors = self.get_neighbours(row, col, scheme=scheme)
                downstream_neighbors = [
                    (dr, dc, direction_idx) for dr, dc, direction_idx in neighbors if self.l[row, col] - self.l[row+dr, col+dc] > self.depth_tolerance
                ]

                # If no downstream neighbors, move to the next cell
                if len(downstream_neighbors) == 0:
                    continue

                # Compute available storage volumes
                dV_0i = np.zeros(len(downstream_neighbors), dtype=np.float64)
                dV_total = 0
                for i, (dr, dc, _) in enumerate(downstream_neighbors):
                    r, c = row + dr, col + dc
                    dl = l0 - self.l[r, c]
                    if abs(r-row) == abs(c - col):
                        dV_0i[i] = (self.dx**2 / np.sqrt(2)) * max(dl, 0) # Diagonal cell (Moore's neighbourhood)
                    else:
                        dV_0i[i] = self.dx**2 * max(dl, 0)

                    # if self.porous_bc[r, c]:
                    #     dV_0i = dV_0i * (1 - self.porous_bc[r, c])
                    
                    dV_total += dV_0i[i]

                dV_min = np.min(dV_0i)
                dV_max = np.max(dV_0i)

                # Compute weights for downstream neighbours and central cell
                w_i = np.zeros(len(downstream_neighbors), dtype=np.float64)
                for i in range(len(downstream_neighbors)):
                    w_i[i] = dV_0i[i] / (dV_total + dV_min)
                w_0 = dV_min / (dV_total + dV_min)

                # Compute total intercellular volume
                max_weight_idx = np.argmax(w_i)
                dr_M, dc_M, _ = downstream_neighbors[max_weight_idx]
                r_M, c_M = row + dr_M, col + dc_M

                # maximum possible intercellular velocity (v_M)
                dl0_M = l0 - self.l[r_M, c_M] # water level of cell with largest weight

                # distance between the centre of the central cell and the centre of cell with the largest weight
                if abs(r_M - row) == abs(c_M - col):
                    dx0_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                else:
                    dx0_M = self.dx

                v_critical = np.sqrt(d0 * 9.81)  # Critical velocity
                v_manning = (1 / self.n) * (d0**(2 / 3)) * np.sqrt(dl0_M / dx0_M)  # Manning velocity
                v_M = min(v_critical, v_manning)

                # max possible intercellular volume (I_M)
                # Length of the cell edge with largest weight
                if abs(r_M - row) == abs(c_M - col):
                    de_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                else:
                    de_M = self.dx
                I_M = v_M * d0 * self.dt * de_M

                # Compute total intercellular volume (I_total)
                A0 = self.dx**2
                w_M = w_i[max_weight_idx]
                new_I_total[row, col] = min(
                    d0 * A0,
                    I_M / w_M,
                    dV_min + self.I_total[row, col]
                )

                if self.porous_bc[row, col] > 0:
                    new_I_total = new_I_total * (1 - self.porous_bc[row, col])

                # Distribute intercellular volume to neighbors
                for i, (dr, dc, direction_idx) in enumerate(downstream_neighbors):
                    r, c = row + dr, col + dc

                    I_i = w_i[i] * new_I_total[row, col]
                    I_ij[row, col, direction_idx] = I_i
                    new_d[r,c] += I_i / A0 # update neighbours water depth
                    if new_d[r,c] < 0:
                        new_d[r,c] = 0

                # Update water depth for the next time step
                I_i_total = sum(I_ij[row, col, :])
                new_d[row, col] = new_d[row, col] - I_i_total / A0
                
                if new_d[row, col] < 0:
                    new_d[row, col] = 0

        return new_d, new_I_total, I_ij

    def apply_boundary_conditions(self, new_d):
        new_new_d = np.copy(new_d)

        A = self.dx**2

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                
                # Open outlet
                if self.open_out_bc[row, col]:
                    new_new_d[row, col] = self.d[row, col]

                # Volumetric flow rate inflow
                new_new_d[row, col] += self.vfr_in_bc[row, col] * self.dt / A

                # Volumetric flow rate outflow
                new_new_d[row, col] -= self.vfr_out_bc[row, col] * self.dt / A

                # Wall
                if self.wall_bc[row, col]:
                    new_new_d[row, col] = 0

                # Clamp at 0
                if new_new_d[row, col] < 0:
                    new_new_d[row, col] = 0


        return new_new_d


    def compute_intercellular_velocity(self, I_ij):
        vel = np.zeros((self.grid_shape[0], self.grid_shape[1], 2))

        scheme = "moore" if I_ij.shape[2] == 8 else "von_neumann"

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                # Get neighbours
                neighbours = self.get_neighbours(row, col, scheme)

                u, v = 0, 0 # components of velocity vector

                if self.d[row, col] > self.depth_tolerance:
                    for dr, dc, direction_idx in neighbours:
                        r, c = row + dr, col + dc

                        # Average depth
                        d_avg = 0.5 * (self.d[row, col] + self.d[r, c])

                        # Length of the neighbour cell edge
                        if abs(r - row) == abs(c - col):
                            de = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                        else:
                            de = self.dx

                        # Compute intercellular velocities
                        v_i = I_ij[row, col, direction_idx] / (d_avg * de * self.dt)

                        # Add velocity contribution to components
                        u += v_i * dc
                        v += v_i * dr

                vel[row, col, 0] = u
                vel[row, col, 1] = v

        return vel


    def compute_hydraulic_gradient(self, row, col, scheme):
        """
        Compute the hydraulic slope (S) for a given cell based on its neighbors.

        Parameters:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            float: Hydraulic slope (S) for the cell.
        """

        max_slope = 0.0
        neighbors = self.get_neighbours(row, col, scheme=scheme)  # Use Moore or Von Neumann scheme

        l0 = self.l[row, col]  # Water level in the central cell
        for dr, dc, _ in neighbors:
            r, c = row + dr, col + dc

            l_i = self.l[r, c]  # Water level in the neighbour
            dx = self.dx * np.sqrt(2) if abs(r - row) == abs(c - col) else self.dx  # Adjust for diagonal
            slope = abs(l0 - l_i) / dx
            max_slope = max(max_slope, slope)

        return max_slope


    def update_timestep(self, max_dt, scheme, slope_tolerance=0.01):
        """
        Update the time step based on grid properties, Manning's formula, and slope.

        Parameters:
            slope_tolerance (float): Minimum slope (S) to consider for stability.

        Returns:
            float: Updated time step (dt).
        """
        min_dt = float(max_dt)  # Initialize minimum time step
        n = self.n  # Manning's roughness coefficient
        dx = self.dx  # Grid cell size (assumed square)

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                # Compute hydraulic radius (R) and slope (S)
                d = self.d[row, col]  # Water depth in the cell
                if d <= self.depth_tolerance:
                    continue  # Skip dry cells

                R = d  # Hydraulic radius approximated by water depth
                S = self.compute_hydraulic_gradient(row, col, scheme)

                if S > slope_tolerance:  # Only consider slopes above the tolerance
                    # Compute time step for this cell
                    dt_cell = (dx**2 / 4) * ((2 * n / R**(5/3)) * S**0.5)
                    min_dt = min(min_dt, dt_cell)  # Update minimum time step
                else:
                    # print(S, slope_tolerance)
                    pass
        
        return min_dt


    def run_simulation(self, dt, max_dt, total_time, output_interval, scheme="von_neumann"):
        """
        Run the WCA2D simulation for a given total time.
        
        Parameters:
            total_time (float): Total simulation time.
            output_interval (float): Time interval to save and output results.
        """
        self.dt = dt
        time = 0
        update_time = output_interval

        ds = [self.d]
        vs = []

        while time < total_time:
            new_d, new_I_total, I_ij = self.compute_intercellular_volume(scheme=scheme)

            new_d = self.apply_boundary_conditions(new_d)

            self.d = new_d
            self.l = self.z + self.d
            self.I_total = new_I_total

            ds.append(self.d.copy())

            time += self.dt
            if time >= update_time:

                vel = self.compute_intercellular_velocity(I_ij)
                vs.append(vel)
                self.dt = self.update_timestep(max_dt, scheme=scheme)
                # print(self.dt)
                
                update_time += output_interval

        return ds, vs
