#!/usr/bin/env python

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None
pass


class PhiCut(object):
    @classmethod
    def plane(cls, phi):
        return np.array( [np.sin(np.pi*phi), -np.cos(np.pi*phi), 0, 0 ], dtype=np.float32 )

    def __init__(self, phiStart, phiDelta ):
        phi0 = phiStart
        phi1 = phiStart + phiDelta

        cosPhi0, sinPhi0 = np.cos(phi0*np.pi), np.sin(phi0*np.pi)
        cosPhi1, sinPhi1 = np.cos(phi1*np.pi), np.sin(phi1*np.pi)
        n0 = np.array( [  sinPhi0, -cosPhi0, 0. ]  )
        n1 = np.array( [ -sinPhi1,  cosPhi1, 0. ]  )
        self.n0 = n0 
        self.n1 = n1 

    def intersect(self, ray_origin, ray_direction, t_min ):
        """
        These are currently full planes need to disqualify half of them. 
        What exactly is the convention for the allowed intersects ?
        """
        n0 = self.n0
        n1 = self.n1

        dn0 = np.dot(ray_direction, n0 )
        on0 = np.dot(ray_origin,    n0 )
        t0 = -on0/dn0 

        dn1 = np.dot(ray_direction, n1 )
        on1 = np.dot(ray_origin,    n1 )
        t1 = -on1/dn1 

        t_near = min(t0,t1)
        t_far  = max(t0,t1)

        #t_cand = t_near > t_min  ?  t_near : ( t_far > t_min ? t_far : t_min ) ;
        t_cand = t_min  
        if t_near > t_min:
            t_cand = t_near
        else:
            if t_far > t_min:
                t_cand = t_far
            else:     
                t_cand = t_min
            pass
        pass

        n1_hit = t_cand == t1 

        valid_intersect = t_cand > t_min 
        if valid_intersect: 
            isect = np.zeros(4)
            isect[:3]= n1 if n1_hit else n0
            isect[3] = t_cand
        else:
            isect = None
        pass
        return isect 
  
         



def check_normals():
    #phi = np.linspace(0,2,100)
    phi = np.linspace(0,2,10)

    planes = np.zeros( (len(phi), 4), dtype=np.float32 )
    pos = np.zeros( (len(phi), 3), dtype=np.float32 )

    for i in range(len(phi)):
        planes[i] = PhiCut.plane(phi[i])
        pos[i] = (10*np.cos(phi[i]*np.pi), 10*np.sin(phi[i]*np.pi), 0 )
    pass

    size = np.array([1280, 720])
    pl = pv.Plotter(window_size=size*2 )
    #pl.camera.ParallelProjectionOn()
    pl.view_xy() 
    pl.add_arrows( pos,  planes[:,:3],  color="red" )   # suitable for phi0
    pl.add_arrows( pos, -planes[:,:3],  color="blue" )  # suitable for phi1

    look = (0,0,0)
    up = (0,1,0)
    eye = (0,0, 10)

    pl.set_focus(    look )
    pl.set_viewup(   up )
    pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
    pl.camera.Zoom(1)
    pl.show_grid()
    cp = pl.show()



class Scanner(object):
    def __init__(self, geom ):
        self.geom = geom 

    def vertical(self, n, t_min=0 ):
        geom = self.geom
        ipos = np.zeros( (n*2, 3) ) 
        iray = np.zeros( (n*2, 2, 3) ) 

        for i in range(n):
            j = i - n/2 
            ray_origin    = np.array( [  j*0.1,   0, 0 ] )    
            ray_direction = np.array( [     0,    1 if j > 0 else -1 , 0 ] )    

            iray[i,0] = ray_origin
            iray[i,1] = ray_direction

            isect = geom.intersect( ray_origin, ray_direction, t_min )
            if not isect is None:
                ipos[i] = isect[3]*ray_direction + ray_origin 
            pass
        pass
        for i in range(n):
            j = i - n/2 

            ray_origin    = np.array( [     0,   j*0.1, 0 ] )    
            ray_direction = np.array( [     1 if j > 0 else -1,     0, 0 ] )    

            iray[i+n,0] = ray_origin
            iray[i+n,1] = ray_direction

            isect = geom.intersect( ray_origin, ray_direction, t_min )
            if not isect is None:
                ipos[i+n] = isect[3]*ray_direction + ray_origin 
            pass
        pass
        self.ipos = ipos
        self.iray = iray
    


if __name__ == '__main__':
    pc = PhiCut( 0.25, 0.1 )
    sc = Scanner(pc)
    sc.vertical(100)

    
    size = np.array([1280, 720])
    pl = pv.Plotter(window_size=size*2 )
    #pl.camera.ParallelProjectionOn()
    pl.view_xy() 
    pl.add_points( sc.ipos,  color="white" )  
    pl.add_points( sc.iray[:,0],  color="red" )  


    look = (0,0,0)
    up = (0,1,0)
    eye = (0,0, 10)

    pl.set_focus(    look )
    pl.set_viewup(   up )
    pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
    pl.camera.Zoom(1)
    pl.show_grid()
    cp = pl.show()














