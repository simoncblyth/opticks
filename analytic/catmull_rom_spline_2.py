#!/usr/bin/env python
"""
catmull_rom_spline_2.py
==========================

This 2nd implementation factors out the calculation of the CatMull Rom interpolation 
weights : as they are the same for every segment. 

This means that the interpolation reduces to a single matrix multiply
for each segment. 

"""

import numpy as np, matplotlib as mp
SIZE = np.array([1280, 720])

def catmull_rom_Mu_( u, a=0.5 ):
    """
    Derivation of the interpolation basis functions:

    * https://www.cs.cmu.edu/~fp/courses/graphics/asst5/catmullRom.pdf
    * ~/opticks_refs/derivation_catmullRom_spline.pdf

    """
    return np.array([  
             -a*u + 2*a*u*u -a*u*u*u, 
             1+(a-3)*u*u+(2-a)*u*u*u,
             a*u+(3-2*a)*u*u+(a-2)*u*u*u,
             -a*u*u+a*u*u*u ])  


def catmull_rom_Mu( N=100, a=0.5 ):
    """
    :param N: number of interpolated points for each segment
    :param a: "tension"  0.5 is standard for Catmull-Rom
    :return interpolation weights: shape (N, 4)  

    Every segment (which is controlled by a group of 4 control points) 
    needs to use the same Mu for its interpolation so invoke this once only 
    before the loop over segments. 
    """
    return catmull_rom_Mu_(np.linspace(0,1,N), a ).T

def circle_points(n=16):
    angles = np.arange(0, 2*np.pi, 2*np.pi/n) # NB does not get to 2pi to avoid repeated point    
    points = np.zeros( (len(angles), 3) )
    points[:,0] = np.cos(angles)
    points[:,1] = np.sin(angles)
    return points

def square_points():
    return np.array([ [0,0,0], [1,0,0], [0,1,0], [1,1,0] ])  

class Plot(object):
    def __init__(self, points, xpp, title):
        fig, ax = mp.pyplot.subplots(figsize=SIZE/100.) 
        fig.suptitle(title)
        ax.set_aspect('equal')
        ax.plot( xpp[:,0], xpp[:,1], label="xpp"  )
        ax.scatter( points[:,0], points[:,1], label="points" )
        ax.legend()
        fig.show()

if __name__ == '__main__':
    points = circle_points(16)
    assert len(points) >= 4 
    numseg = len(points)-3     # sliding window of 4 control points for each segment  
    looped = True
    if looped: numseg += 3    

    N = 100                           # number of points to interpolate each segment into 
    Mu = catmull_rom_Mu( N, a=0.5 )   # interpolation weights to apply to each group of 4 control points
    print(Mu.shape)

    xpp = np.zeros( (numseg*N,3) )  
    for i in range(numseg):
        ## p: (4,3) sliding sub-array of 4 control points that wraps around at the end  
        p = np.take( points, np.arange(i,i+4), mode='wrap', axis=0 ) 
        xpp[i*N:(i+1)*N] = np.dot( Mu, p ) 
    pass
    Plot(points, xpp, "analytic/catmull_rom_spline_2.py" )

