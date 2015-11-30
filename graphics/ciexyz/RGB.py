#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from space import Space

class RGB(object):
    """
    See sRGB.py for intro to color spaces starting from primary Chromaticities
    This starts from the matrices
    """
    prim = np.array([
           [1,1,1],
           [1,0,0],
           [0,1,0],
           [0,0,1],
           [0,1,1],
           [1,0,1],
           [1,1,0]], dtype=np.float64)

    names = "W R G B C M Y".split()

    @classmethod 
    def spaces(cls):
        return Space.s.keys()

    def __init__(self, name="sRGB/D65" ):
        x2r, r2x = Space.space(name)

        x2r_4 = np.identity(4)
        x2r_4[:3,:3] = x2r

        r2x_4 = np.identity(4)
        r2x_4[:3,:3] = r2x


        XYZ = np.dot(r2x, self.prim.T).T
        xyz = XYZ/np.repeat(np.sum(XYZ, axis=1),3).reshape(-1,3)

        self.x2r = x2r 
        self.r2x = r2x 
        self.x2r_4 = x2r_4 
        self.r2x_4 = r2x_4 

        self.XYZ = XYZ
        self.xyz = xyz

    def table(self):
        for i in range(len(self.XYZ)):
            print "%3s : rgb %15s XYZ %15s sum(%4.2f)  xyz %15s " % ( 
                  self.names[i], 
                  self.prim[i], 
                  self.XYZ[i],  
                  self.XYZ[i].sum(), 
                  self.xyz[i] ) 

    def plot(self):
        xyz = self.xyz
        plt.scatter( xyz[:,0], xyz[:,1] )
        for i in range(len(self.names)):
            plt.annotate(self.names[i], xy = xyz[i,:2], xytext = (0.5, 0.5), textcoords = 'offset points')



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    rgb = RGB()
    rgb.table()




