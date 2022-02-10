#!/usr/bin/env python

import numpy as np
import pyvista as pv
SIZE = np.array([1280, 720])



def make_stage(sz):
     stage = np.zeros( (6,3), dtype=np.float32 )
     stage[0] = (-sz,   0,   0)
     stage[1] = ( sz,   0,   0)
     stage[2] = (  0, -sz,   0)
     stage[3] = (  0,  sz,   0)
     stage[4] = (  0,   0, -sz)
     stage[5] = (  0,   0,  sz)
     return stage 


if __name__ == '__main__':

     path = "/tmp/CSGQueryTest.npy"
     isect = np.load(path)


     norm, t       = isect[:,0,:3], isect[:,0,3]
     pos, sd       = isect[:,1,:3], isect[:,1,3]
     ray_origin    = isect[:,2,:3] 
     ray_direction = isect[:,3,:3] 


     ll = np.zeros( (len(t), 2, 3), dtype=np.float32 )
     ll[:,0] = ray_origin
     ll[:,1] = pos


     pl = pv.Plotter(window_size=SIZE*2 )
     
     stg = make_stage(10)
     pl.add_points( stg, color="yellow" )

     pl.add_points( pos, color="white" ) 
     pl.add_arrows( pos, norm , color="red", mag=1 )
     pl.add_points( ray_origin, color="magenta", point_size=16.0 )

     for i in range(len(ll)):
         pl.add_lines( ll[i].reshape(-1,3), color="blue" )
     pass  

     look = np.array( [0,0,0], dtype=np.float32 )
     up = np.array( [0,0,1], dtype=np.float32 )
     eye = np.array( [0,-20,0], dtype=np.float32 )
     zoom = 1 

     pl.set_focus(    look )
     pl.set_viewup(   up )
     pl.set_position( eye, reset=False )  
     pl.camera.Zoom(zoom)

     pl.show_grid()

     cp = pl.show()





