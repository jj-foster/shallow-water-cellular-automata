import numpy as np

from visualise import *

class WCA2D:
    def __init__(self, grid_shape, z, d, dx, dt, depth_tolerance, n):
        self.grid_shape = grid_shape
        self.z = z
        self.d = d
        self.l = z + d
        self.dx = dx
        self.dt = dt
        self.depth_tolerance = depth_tolerance
        self.n = n
        self.I_total = np.zeros_like(self.d)
    
    def get_neighbours(self, row, col, scheme):
        """
        Get neighbors for a given cell in the grid based on the chosen scheme.
        
        Parameters:
            row (int): Row index of the central cell.
            col (int): Column index of the central cell.
            scheme (str): The neighborhood scheme to use ('von_neumann' or 'moore').
            
        Returns:
            List of tuples (neighbor_row, neighbor_col) representing neighbor indices.
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
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_shape[0] and 0 <= c < self.grid_shape[1]:
                neighbors.append((r, c))
        
        return neighbors
    
    def compute_intercellular_volume(self, scheme):
        """
        Compute intercellular volumes for all cells in the grid.
        Updates water depth for the next time step.
        """
        new_d = np.copy(self.d)  # To store updated water depths
        new_I_total = np.zeros_like(self.I_total)  # To store updated total intercellular volumes

        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                # Get central cell properties
                l0 = self.l[row, col]
                d0 = self.d[row, col]

                # Identify downstream neighbors
                neighbors = self.get_neighbours(row, col, scheme=scheme)
                downstream_neighbors = [
                    (r, c) for r, c in neighbors if self.l[row, col] - self.l[r, c] > self.depth_tolerance
                ]

                # If no downstream neighbors, move to the next cell
                if len(downstream_neighbors) == 0:
                    continue

                # Compute available storage volumes
                dV_0i = np.zeros(len(downstream_neighbors), dtype=np.float64)
                dV_total = 0
                for i, (r, c) in enumerate(downstream_neighbors):
                    dl = l0 - self.l[r, c]
                    if abs(r-row) == abs(c - col):
                        dV_0i[i] = (self.dx**2 / np.sqrt(2)) * max(dl, 0) # Diagonal cell (Moore's neighbourhood)
                    else:
                        dV_0i[i] = self.dx**2 * max(dl, 0)
                    dV_total += dV_0i[i]

                dV_min = np.min(dV_0i)
                dV_max = np.max(dV_0i)

                # Compute weights for downstream neighbours and central cell
                w_i = np.zeros(len(downstream_neighbors), dtype=np.float64)
                for i, (r, c) in enumerate(downstream_neighbors):
                    w_i[i] = dV_0i[i] / (dV_total + dV_min)
                w_0 = dV_min / (dV_total + dV_min)

                # Compute total intercellular volume
                max_weight_idx = np.argmax(w_i)
                r_M, c_M = downstream_neighbors[max_weight_idx]

                # maximum possible intercellular velocity (v_M)
                dl0_M = l0 - self.l[r_M, c_M] # water level of cell with largest weight
                # distance between the centre of the central cell and the centre of cell with the largest weight
                # dx0_M = self.dx 
                if abs(r_M - row) == abs(c_M - col):
                    dx0_M = self.dx * np.sqrt(2) # diagonal cell (Moore's neighbourhood)
                else:
                    dx0_M = self.dx

                v_critical = np.sqrt(d0 * 9.81)  # Critical velocity
                v_manning = (1 / self.n) * (d0**(2 / 3)) * np.sqrt(dl0_M / dx0_M)  # Manning velocity
                v_M = min(v_critical, v_manning)

                # max possible intercellular volume (I_M)
                # Length of the cell edge with largest weight
                # de_M = self.dx  
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

                # Distribute intercellular volume to neighbors
                I_i = np.zeros(len(downstream_neighbors), dtype=np.float64)
                for i, (r, c) in enumerate(downstream_neighbors):
                    I_i[i] = w_i[i] * new_I_total[row, col]
                    new_d[r,c] += I_i[i] / A0 # update neighbours water depth

                # Update water depth for the next time step
                I_i_total = sum(I_i)
                new_d[row, col] = new_d[row, col] - I_i_total / A0

        return new_d, new_I_total
    
    # def compute_intercellular_velocity(self, row, col, downstream_neighbours, )
    
    def run_simulation(self, total_time, output_interval, scheme="von_neumann"):
        """
        Run the WCA2D simulation for a given total time.
        
        Parameters:
            total_time (float): Total simulation time.
            output_interval (float): Time interval to save and output results.
        """
        time = 0

        ds = [self.d]


        while time < total_time:
            new_d, new_I_total = self.compute_intercellular_volume(scheme=scheme)
            self.d = new_d
            self.l = self.z + self.d
            self.I_total = new_I_total

            
            ds.append(self.d.copy())

            time += self.dt
            if time % output_interval < self.dt:
                pass

        return ds

if __name__ == "__main__":
    grid_shape = (5, 5)
    
    z = np.zeros(grid_shape)
    z = np.array([
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2]])

    d = np.full(grid_shape, 0.0)
    d[0, 0] = 1.0

    depth_tolerance = 0.01
    n = 0.03

    total_time = 10.0
    dt = 0.1
    output_interval = 0.5

    wca = WCA2D(grid_shape, z, d, dx=1.0, dt=dt, depth_tolerance=depth_tolerance, n=n)
    ds = wca.run_simulation(total_time=10.0, output_interval=1.0, scheme="moore")

    # visualize_cell_parameter(ds, zlabel='Water Depth', interval=500)
    visualize_water_depth_3d(ds,interval=1000)