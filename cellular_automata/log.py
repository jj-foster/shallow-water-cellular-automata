import numpy as np

class Log:
    def __init__(self):
        self.time = []
        self.update_time = []
        self.dt = []
        self.d = []
        self.l = []
        self.vel = []

        self.grid_shape = None

    # def speed(self):
    #     vel = self.to_2D(self.vel, self.grid_shape[0], self.grid_shape[1])
    #     return [np.sqrt(v[:,:,0]**2 + v[:,:,1]**2) for v in vel]

    def speed(self):
        return self.to_2D(self.vel, self.grid_shape[0], self.grid_shape[1])
    
    # def u(self, scale=1):
    #     vel = self.to_2D(self.vel, self.grid_shape[0], self.grid_shape[1])
    #     return [v[:,:,0] * scale for v in vel]
    
    # def v(self, scale=1):
    #     vel = self.to_2D(self.vel, self.grid_shape[0], self.grid_shape[1])
    #     return [v[:,:,1] * scale for v in vel]
    
    def mv_avg(self, var, window_size):
        """
        Compute a time-averaged version of a list of numpy arrays.
        
        Parameters:
            fields (list of np.ndarray): Time-dependent field variables.
            window_size (int): Number of time steps to average over.
        
        Returns:
            list of np.ndarray: Time-averaged field variables.
        """
        avg_vars = []
        n = len(var)

        for t in range(n):
            # Define the window range
            start = max(0, t - window_size + 1)
            end = t + 1  # Include the current time step
            # Compute the mean over the window
            avg_var = np.mean(var[start:end], axis=0)
            avg_vars.append(avg_var)
        
        return avg_vars 
    
    @staticmethod
    def to_2D(var, rows, cols):
        """
        Convert a 1D array to a 2D array.
        
        Parameters:
            var (np.ndarray): 1D array.
            rows (int): Number of rows in the 2D array.
            cols (int): Number of columns in the 2D array.
        
        Returns:
            np.ndarray: 2D array.
        """
        return [np.array(arr).reshape(rows, cols) for arr in var]

