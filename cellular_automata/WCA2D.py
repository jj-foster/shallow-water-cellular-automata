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
        self.I_total = np.zeros_like(self.d)

        self.wall_bc = self.flatten(wall_bc.astype(float))
        self.vfr_in_bc = self.flatten(vfr_in_bc.astype(float))
        self.vfr_out_bc = self.flatten(vfr_out_bc.astype(float))
        self.open_out_bc = self.flatten(open_out_bc.astype(float))
        self.porous_bc = self.flatten(porous_bc.astype(float))

        self.neighbours = None
        self.scheme = None

        self.log = Log()
        self.log.grid_shape = grid_shape

        self.MOORE_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),         (0, 1),
                        (1, -1), (1, 0), (1, 1)]
        self.VON_NEUMANN_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
                for direction_idx, (dr, dc) in enumerate(directions):
                    r, c = row + dr, col + dc

                    y = (-1, direction_idx)
                    if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                        j = r * self.grid_shape[1] + c
                        if not self.wall_bc[i]:
                            y = (j, direction_idx)

                    x.append(y)

                neighbors.append(x)
                i+=1
        
        return neighbors
    

    def compute_intercellular_volume(self, scheme):
        """
        Compute intercellular volumes for all cells in the grid.
        Updates water depth for the next time step.
        """
        new_d = np.copy(self.d)  # To store updated water depths
        new_I_total = np.zeros_like(self.I_total)  # To store updated total intercellular volumes

        max_neighbours = 8 if scheme == "moore" else 4
        I_ij = np.zeros((self.grid_shape[0] * self.grid_shape[1], max_neighbours), dtype=np.float64)

        for i in range(self.l.size):

            if self.wall_bc[i]:
                continue

            # Get central cell properties
            l0 = self.l[i]
            d0 = self.d[i]

            # Identify downstream neighbors
            downstream_neighbors = [
                (j, direction_idx) for j, direction_idx in self.neighbours[i] \
                    if self.l[i] - self.l[j] > self.depth_tolerance and j != -1
            ]
            
            # If no downstream neighbors, move to the next cell
            if len(downstream_neighbors) == 0:
                continue

            # Compute available storage volumes
            dV_0j = np.zeros(len(downstream_neighbors), dtype=np.float64)
            dV_total = 0
            for k,(j,_) in enumerate(downstream_neighbors):
                dl = l0 - self.l[j]
                if scheme == "moore":
                    dV_0j[k] = (self.dx**2 / np.sqrt(2)) * max(dl, 0) # Diagonal cell (Moore's neighbourhood)
                else:
                    dV_0j[k] = self.dx**2 * max(dl, 0)

                # if self.porous_bc[r, c]:
                #     dV_0i = dV_0i * (1 - self.porous_bc[r, c])
                
                dV_total += dV_0j[k]

            dV_min = np.min(dV_0j)
            dV_max = np.max(dV_0j)

            # Compute weights for downstream neighbours and central cell
            w_j = np.zeros(len(downstream_neighbors), dtype=np.float64)
            for j in range(len(downstream_neighbors)):
                w_j[j] = dV_0j[j] / (dV_total + dV_min)
            w_0 = dV_min / (dV_total + dV_min)

            # Compute total intercellular volume
            max_weight_idx = np.argmax(w_j)
            j_m, _ = downstream_neighbors[max_weight_idx]

            # maximum possible intercellular velocity (v_M)
            dl0_M = l0 - self.l[j_m] # water level of cell with largest weight

            # distance between the centre of the central cell and the centre of cell with the largest weight
            if scheme=="moore":
                dx0_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
            else:
                dx0_M = self.dx

            v_critical = np.sqrt(d0 * 9.81)  # Critical velocity
            v_manning = (1 / self.n) * (d0**(2 / 3)) * np.sqrt(dl0_M / dx0_M)  # Manning velocity
            v_M = min(v_critical, v_manning)

            # max possible intercellular volume (I_M)
            # Length of the cell edge with largest weight
            if scheme == "moore":
                de_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
            else:
                de_M = self.dx
            I_M = v_M * d0 * self.dt * de_M

            # Compute total intercellular volume (I_total)
            w_M = w_j[max_weight_idx]
            new_I_total[i] = min(
                d0 * self.A0,
                I_M / w_M,
                dV_min + self.I_total[i]
            )

            if self.porous_bc[i] > 0:
                new_I_total = new_I_total * (1 - self.porous_bc[i])

            # Distribute intercellular volume to neighbors
            for k, (j, direction_idx) in enumerate(downstream_neighbors):

                I_i = w_j[k] * new_I_total[i]
                I_ij[i, direction_idx] = I_i
                new_d[j] += I_i / self.A0 # update neighbours water depth
                if new_d[j] < 0:
                    new_d[j] = 0

            # Update water depth for the next time step
            I_i_total = sum(I_ij[i, :])
            new_d[i] = new_d[i] - I_i_total / self.A0
            
            if new_d[i] < 0:
                new_d[i] = 0

            # i = i + 1

        return new_d, new_I_total, I_ij

    def apply_boundary_conditions(self, new_d):
        new_new_d = np.copy(new_d)

        A = self.A0

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                
                # Open outlet
                if self.open_out_bc[row, col]:
                    if new_new_d[row, col] > self.d[row, col]:
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


    def compute_intercellular_velocity(self, I_ij, scheme):
        vel = np.zeros(self.grid_shape[0] * self.grid_shape[1])

        if scheme == "von_neumann":
            directions = self.VON_NEUMANN_DIRECTIONS
        else:
            directions = self.MOORE_DIRECTIONS

        for i in range(self.l.size):
            # Get neighbours
            neighbours = self.neighbours[i]

            u, v = 0, 0 # components of velocity vector

            if self.d[i] > self.depth_tolerance:
                for j, direction_idx in neighbours:

                    # Average depth
                    d_avg = 0.5 * (self.d[i] + self.d[j])

                    # Length of the neighbour cell edge
                    if scheme == "moore":
                        de = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                    else:
                        de = self.dx

                    # Compute intercellular velocities
                    v_i = I_ij[i, direction_idx] / (d_avg * de * self.dt)

                    # Add velocity contribution to components
                    # u += v_i * directions[direction_idx][1]
                    # v += v_i * directions[direction_idx][0]
                    vel[i] = v_i

            # vel[i, 0] = u
            # vel[i, 1] = v

        return vel


    def compute_hydraulic_gradient(self, i, scheme):
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
        for j, idx in neighbors:

            l_j = self.l[j]  # Water level in the neighbour
            if scheme == "moore":
                dx = self.dx * np.sqrt(2)  # Diagonal cell (Moore's neighbourhood)
            else:
                dx = self.dx  # Rectangular cell (von Neumann neighbourhood)
            
            slope = abs(l0 - l_j) / dx
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
                S = self.compute_hydraulic_gradient(i, scheme)

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
        time = 0
        update_time = output_interval

        self.neighbours = self.get_neighbours(scheme=scheme)

        self.log.time = [time]
        self.log.update_time = [time]
        self.log.d = [self.d]
        self.log.l = [self.l]
        self.log.dt = [self.dt]
        self.log.vel = [np.zeros(self.grid_shape[0] * self.grid_shape[1])]

        while time < total_time:
            new_d, new_I_total, I_ij = self.compute_intercellular_volume(scheme=scheme)

            # new_d = self.apply_boundary_conditions(new_d)

            self.d = new_d
            self.l = self.z + self.d
            self.I_total = new_I_total

            self.log.d.append(self.d)
            self.log.dt.append(self.dt)
            self.log.l.append(self.l)

            time += self.dt
            self.log.time = time

            if time >= update_time:

                vel = self.compute_intercellular_velocity(I_ij, scheme)
                self.log.vel.append(vel)
                self.dt = self.update_timestep(max_dt, scheme=scheme)
                # print(self.dt)
                
                update_time += output_interval
                self.log.update_time = time

        return None
