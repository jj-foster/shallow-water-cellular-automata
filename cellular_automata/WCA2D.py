import numpy as np
from cellular_automata.log import Log

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
        self.z = self.flatten(z.astype(float))
        self.d = self.flatten(d.astype(float))
        self.l = self.z + self.d
        self.dx = dx
        self.A0 = self.dx**2
        self.depth_tolerance = depth_tolerance
        self.n = n

        self.wall_bc = self.flatten(wall_bc.astype(float))
        self.vfr_in_bc = self.flatten(vfr_in_bc.astype(float))
        self.vfr_out_bc = self.flatten(vfr_out_bc.astype(float))
        self.open_out_bc = self.flatten(open_out_bc.astype(float))
        self.porous_bc = self.flatten(porous_bc.astype(float))

        self.neighbours = None
        self.outflux1 = None # outflux for current timestep
        self.outflux2 = None # outflux for previous timestep
        self.scheme = None

        self.log = Log()
        self.log.grid_shape = grid_shape

        self.MOORE_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.VON_NEUMANN_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def flatten(self, array):
        return array.flatten()
    
    def get_neighbours(self, scheme):
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
            directions = self.VON_NEUMANN_DIRECTIONS
        elif scheme == "moore":
            # Moore neighbors (All surrounding cells)
            directions = self.MOORE_DIRECTIONS
        else:
            raise ValueError(f"Unsupported neighborhood scheme: {scheme}")
        
        neighbors = []
        i = 0
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                x = []
                for (dr, dc) in directions:
                    r, c = row + dr, col + dc

                    y = -1
                    if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                        j = r * self.grid_shape[1] + c
                        if not self.wall_bc[i]:
                            y = j

                    x.append(y)

                neighbors.append(x)
                i+=1
        
        return neighbors
    
    def compute_outflow(self, i, ratio_dt):
        
        if self.wall_bc[i]:
            self.outflux1[i,:] = 0
            return None

        # Get central cell properties
        l0 = self.l[i]
        d0 = self.d[i]

        # Identify downstream neighbors
        no_skip = 0
        downstream_neighbors = np.zeros_like(self.neighbours[i])
        for j, ni in enumerate(self.neighbours[i]):
            if ni != -1 and self.l[i] - self.l[ni] > self.depth_tolerance:
                downstream_neighbors[j] = 1
                no_skip = 1
        
        # If no downstream neighbors, move to the next cell
        if no_skip == 0:
            self.outflux1[i,:] = 0
            return None

        # Compute available storage volumes
        dV_0j = np.zeros(len(downstream_neighbors), dtype=np.float64)
        dV_total = 0
        dV_min = 1000000
        I_total = 0 # total flux out of central cell from last timestep
        for j,ni in enumerate(self.neighbours[i]):
            if downstream_neighbors[j] == 0:
                continue

            dl = l0 - self.l[ni]
            if self.scheme == "moore":
                dV_0j[j] = (self.dx**2 / np.sqrt(2)) * max(dl, 0) # Diagonal cell (Moore's neighbourhood)
            else:
                dV_0j[j] = self.dx**2 * max(dl, 0)

            # if dV_0j[j] > 0:
            I_total += self.outflux2[i, j] / ratio_dt

            dV_total += dV_0j[j]
            dV_min = min(dV_min, dV_0j[j])

        # Compute weights for downstream neighbours and central cell
        w_j = np.zeros(len(downstream_neighbors), dtype=np.float64)
        for j,ni in enumerate(self.neighbours[i]):
            if downstream_neighbors[j] == 0:
                continue

            w_j[j] = dV_0j[j] / (dV_total + dV_min)

        
        # Compute total intercellular volume
        max_weight_idx = np.argmax(w_j)
        ni_m = self.neighbours[i][max_weight_idx]

        # maximum possible intercellular velocity (v_M)
        dl0_M = l0 - self.l[ni_m] # water level of cell with largest weight

        # distance between the centre of the central cell and the centre of cell with the largest weight
        if self.scheme=="moore":
            dx0_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
        else:
            dx0_M = self.dx

        v_critical = np.sqrt(d0 * 9.81)  # Critical velocity
        v_manning = (1 / self.n) * (d0**(2 / 3)) * np.sqrt(dl0_M / dx0_M)  # Manning velocity
        v_M = min(v_critical, v_manning)

        # max possible intercellular volume (I_M)
        # Length of the cell edge with largest weight
        if self.scheme == "moore":
            de_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
        else:
            de_M = self.dx
        I_M = v_M * d0 * self.dt * de_M

        # Compute total intercellular volume (I_total)
        w_M = w_j[max_weight_idx]
        new_I_total = min(
            d0 * self.A0,
            I_M / w_M,
            dV_min + I_total
        )

        if self.porous_bc[i] > 0:
            new_I_total = new_I_total * (1 - self.porous_bc[i])

        # Calculate outflux for each downstream neighbour considering weights
        for j,ni in enumerate(self.neighbours[i]):
            if downstream_neighbors[j] == 0:
                continue

            I_ni = w_j[j] * new_I_total
            self.outflux1[i, j] = I_ni

        return None


    def compute_water_depth(self, i):
        """
        Compute intercellular volumes for all cells in the grid.
        Updates water depth for the next time step.
        """

        d = self.d[i].copy()

        # loop through edges of the cell
        flux = 0
        for j, ni in enumerate(self.neighbours[i]):
            # flux leaving cell
            flux -= self.outflux1[i, j]

            # flux entering cell
            if self.scheme == "moore":
                flux += self.outflux1[ni, (j+4)%8]
            else:
                flux += self.outflux1[ni, (j+2)%4]
            

        # Update water depth
        d += flux / self.A0

        ### BOUNDARY CONDITIONS ###

        # Open outlet
        if self.open_out_bc[i]:
            if d > self.d[i]:
                d = self.d[i]

        # Volumetric flow rate inflow
        d += self.vfr_in_bc[i] * self.dt / self.A0

        # Volumetric flow rate outflow
        d -= self.vfr_out_bc[i] * self.dt / self.A0

        # Wall
        if self.wall_bc[i]:
            d = 0
        
        if d < 0:
            d = 0

        self.d[i] = d

        return None


    def compute_intercellular_velocity(self):
        vel = np.zeros(self.grid_shape[0] * self.grid_shape[1])

        for i in range(self.l.size):
            # Get neighbours
            neighbours = self.neighbours[i]

            if self.d[i] > self.depth_tolerance:
                for j, ni in enumerate(neighbours):

                    # Average depth
                    d_avg = 0.5 * (self.d[i] + self.d[ni])

                    # Length of the neighbour cell edge
                    if self.scheme == "moore":
                        de = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                    else:
                        de = self.dx

                    # Compute intercellular velocities
                    v_j = self.outflux1[i, j] / (d_avg * de * self.dt)

                    vel[i] += v_j

        return vel


    def compute_hydraulic_gradient(self, i):
        """
        Compute the hydraulic slope (S) for a given cell based on its neighbors.

        Parameters:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            float: Hydraulic slope (S) for the cell.
        """

        max_slope = 0.0
        neighbors = self.neighbours[i]

        l0 = self.l[i]  # Water level in the central cell
        for ni in neighbors:

            l_j = self.l[ni]  # Water level in the neighbour
            if self.scheme == "moore":
                dx = self.dx * np.sqrt(2)  # Diagonal cell (Moore's neighbourhood)
            else:
                dx = self.dx  # Rectangular cell (von Neumann neighbourhood)
            
            slope = abs(l0 - l_j) / dx
            max_slope = max(max_slope, slope)

        return max_slope


    def update_timestep(self, max_dt, slope_tolerance=0.01):
        """
        Update the time step based on grid properties, Manning's formula, and slope.

        Parameters:
            slope_tolerance (float): Minimum slope (S) to consider for stability.

        Returns:
            float: Updated time step (dt).
        """
        dt = float(max_dt)  # Initialize minimum time step
        n = self.n  # Manning's roughness coefficient
        dx = self.dx  # Grid cell size (assumed square)

        i = 0
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                # Compute hydraulic radius (R) and slope (S)
                d = self.d[i]  # Water depth in the cell
                if d <= self.depth_tolerance:
                    continue  # Skip dry cells

                R = d  # Hydraulic radius approximated by water depth
                S = self.compute_hydraulic_gradient(i)

                if S > slope_tolerance:  # Only consider slopes above the tolerance
                    # Compute time step for this cell
                    dt_cell = (dx**2 / 4) * ((2 *n* S**0.5) / R**(5/3))
                    dt = min(dt, dt_cell)  # Update minimum time step
                else:
                    # print(S, slope_tolerance)
                    pass

                i+=1
        
        return dt


    def run_simulation(self, dt, max_dt, total_time, output_interval, scheme="von_neumann"):
        """
        Run the WCA2D simulation for a given total time.
        
        Parameters:
            total_time (float): Total simulation time.
            output_interval (float): Time interval to save and output results.
        """
        self.dt = dt
        self.prev_dt = dt
        self.ratio_dt = dt / self.prev_dt
        time = 0
        update_time = output_interval

        self.scheme = scheme
        self.neighbours = self.get_neighbours(scheme=scheme)
        max_neighbours = 8 if scheme == "moore" else 4
        self.outflux1 = np.zeros((self.l.size, max_neighbours))
        self.outflux2 = np.zeros((self.l.size, max_neighbours))

        self.log.time = [time]
        self.log.update_time = [time]
        self.log.d = [self.d.copy()]
        self.log.l = [self.l.copy()]
        self.log.dt = [self.dt]
        self.log.vel = [np.zeros(self.grid_shape[0] * self.grid_shape[1])]

        while time < total_time:
            self.outflux2 = np.copy(self.outflux1)
            self.outflux1 = np.zeros((self.l.size, max_neighbours))

            # compute outflow for current timestep. Assigns fluxes to self.outflux1.
            for i in range(self.l.size):
                self.compute_outflow(i, self.ratio_dt)

            # compute water depth from outflux1. Assigns to self.d
            for i in range(self.l.size):
                self.compute_water_depth(i)

            # self.d = new_d
            self.l = self.z + self.d

            self.log.d.append(self.d.copy())
            self.log.dt.append(self.dt)
            self.log.l.append(self.l.copy())

            time += self.dt
            self.log.time = time

            if time >= update_time:

                vel = self.compute_intercellular_velocity()
                self.log.vel.append(vel)
                self.prev_dt = self.dt
                self.dt = self.update_timestep(max_dt)
                # print(self.dt)
                
                update_time += output_interval
                self.log.update_time = time

        return None
