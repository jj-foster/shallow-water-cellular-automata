import numpy as np

class Log:
    def __init__(self):
        self.time = []
        self.update_time = []
        self.dt = []
        self.d = []
        self.vel = []

    def speed(self):
        return [np.sqrt(v[:,:,0]**2 + v[:,:,1]**2) for v in self.vel]
    
    def u(self, scale=1):
        return [v[:,:,0] * scale for v in self.vel]
    
    def v(self, scale=1):
        return [v[:,:,1] * scale for v in self.vel]
    
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

