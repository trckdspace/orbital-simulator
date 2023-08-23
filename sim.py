import torch
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

def random_unit_vectors(shape :tuple):
     u = torch.rand(shape) - 0.5
     u_hat = u/u.square().sum(dim=0).sqrt()
     return u_hat

class OrbitSimulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.u = random_unit_vectors((3,width,height))
        self.v = random_unit_vectors((3,width,height))
        
        self.v = self.v.cross(self.u)
        self.v = self.v/self.v.square().sum(dim=0).sqrt()

        self.t = torch.rand((1,width,height)) * 3.1415 * 2
        #self.c = torch.zero_like(u)
        
        # float perigee = (float(rand() % 500) + (radius_of_earth_km + 160.f));
        # float a = perigee / (1 - e);        // std::sqrt((b * b) / (1 - (e * e)));
        # float b = a * std::sqrt(1 - e * e); // std::sqrt((b * b) / (1 - (e * e)));
        # float f = std::sqrt(a * a - b * b);
        
        radius_earth_kms = 6378.1
        
        e = torch.rand((1, width, height)) * 0.75
        # e = 0.01

        perigee = torch.rand((1,width,height)) * 500 + radius_earth_kms + 160.
        
        a = perigee / (1-e)
        b = a * (1-e.square()).sqrt()
        
        f = (a.square() - b.square()).sqrt()
        
        self.c = f * self.u
        
        self.u = a * self.u
        self.v = b * self.v

        MEU = 3.986004418e5

        self.multiplier = b / (a / MEU).sqrt()
        
        self.x_t = self.c + self.t.cos() * self.u + self.t.sin() * self.v
        
        if torch.has_cuda:
            self.c.to('cuda')
            self.u.to('cuda')
            self.v.to('cuda')
            self.t.to('cuda')
            self.x_t.to('cuda')
    
    def propagate(self,delta_t):
        d = self.x_t.square().sum(dim = 0)
        self.t += self.multiplier * delta_t / d
        self.x_t = self.c + self.t.cos() * self.u + self.t.sin() * self.v
        
if __name__ == "__main__":        
        sim = OrbitSimulator(1000,1000)

        fig = plt.figure()
        ax = plt.axes(projection='3d')


        for i in range(500):
            sim.propagate(0.1)
            #print(sim.x_t.flatten())
            
            # xdata = sim.x_t[0,:,:]
            # ydata = sim.x_t[1,:,:]
            # zdata = sim.x_t[2,:, :]

            # ax.scatter3D(xdata, ydata, zdata)

        ax.scatter3D(0,0,0, c='red')
        plt.show()

