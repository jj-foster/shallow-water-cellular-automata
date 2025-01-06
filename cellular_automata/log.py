import numpy as np

class Log:
    def __init__(self):
        self.time = []
        self.dt = []
        self.d = []
        self.vel = []

    def speed(self):

        return [np.sqrt(v[:,:,0]**2 + v[:,:,1]**2) for v in self.vel]
    
    def u(self):
        return [v[:,:,0] for v in self.vel]
    
    def v(self):
        return [v[:,:,1] for v in self.vel]
