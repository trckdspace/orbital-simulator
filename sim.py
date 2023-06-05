import torch
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


class OrbitSimulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        u = torch.rand((3,width,height))
        self.u = u/u.pow(2).sum(dim=0).sqrt()
        
        v = torch.rand_like(u)
        v = v/v.pow(2).sum(dim=0).sqrt()
        
        v = v.cross(u)
        self.v = v/v.pow(2).sum(dim=0).sqrt()
        self.t = torch.rand((1,width,height))
        self.c = torch.rand_like(u)
        
        # float perigee = (float(rand() % 500) + (radius_of_earth_km + 160.f));
        # float a = perigee / (1 - e);        // std::sqrt((b * b) / (1 - (e * e)));
        # float b = a * std::sqrt(1 - e * e); // std::sqrt((b * b) / (1 - (e * e)));
        # float f = std::sqrt(a * a - b * b);
        
        radius_earth_kms = 6378.1
        
        e = torch.rand((1, width, height))
        perigee = torch.rand((1,width,height)) * 500 + radius_earth_kms + 160.
        
        a = perigee / (1-e)
        b = a * (1-e.square()).sqrt()
        
        f = (a.square() + b.square()).sqrt()
        
        self.c = f * self.u
        self.u = a*self.u
        self.v = b*self.v
        
        self.x_t = self.c + self.t.cos() * self.u + self.t.sin() * self.v
        
        if torch.has_cuda:
            self.c.to('cuda')
            self.u.to('cuda')
            self.v.to('cuda')
            self.t.to('cuda')
            self.x_t.to('cuda')
    
    def propagate(self,delta_t):
        self.t += delta_t
        self.x_t = self.c + self.t.cos() * self.u + self.t.sin() * self.v
        
        
sim = OrbitSimulator(2,2)

fig = plt.figure()
ax = plt.axes(projection='3d')


for i in range(100):
    sim.propagate(0.1)
    print(sim.x_t[:,0,0].squeeze())
    
    xdata = sim.x_t[0,:,:]
    ydata = sim.x_t[1,:,:]
    zdata = sim.x_t[2,:, :]

    ax.scatter3D(xdata, ydata, zdata, c=zdata)

plt.show()

