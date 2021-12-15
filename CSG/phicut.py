#!/usr/bin/env python

import numpy as np
import logging
log = logging.getLogger(__name__)

try:
    import pyvista as pv
except ImportError:
    pv = None
pass


class HalfPlane(object):
    @classmethod
    def phi_quadrant(cls, phi):
        """
        This way of defining the quadrant avoids precision of pi issues. 
        Yes, but its better to use the same definition that can be applied to the 
        intersect positions without having to calculate phi from x,y.
        Problems likely at boundaries between quadrants

                         0.5
                 0.75     |     0.25 
                          |
                          |
               1.0  ------+------- 0.0 , 2.0 
                          |
                          |
                1.25      |     1.75
                         1.5 

                         
           cosPhi < 0  |  cosPhi > 0  
                       |
               -+      |     ++           
               01      |     11     
                       |              sinPhi > 0  
               - - - - +----------------------------   
                       |              sinPhi < 0  
               --      |     +-
               00      |     10

        """
        if phi >= 0 and phi < 0.5:
            quadrant = 3  
        elif phi >= 0.5 and phi < 1.0:
            quadrant = 1
        elif phi >= 1.0 and phi < 1.5:
            quadrant = 0
        elif phi >= 1.5 and phi <= 2.0:
            quadrant = 2
        else:
            quadrant = -1
        pass
        return quadrant
        
    @classmethod
    def xy_quadrant(cls, x, y ):
        xpos = int(x >= 0.)
        ypos = int(y >= 0.)
        return 2*xpos + ypos 

    @classmethod
    def test_quadrant(cls, n=17):
        """
        In [3]: np.linspace(0,2,17)
        Out[3]: array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   , 1.125, 1.25 , 1.375, 1.5  , 1.625, 1.75 , 1.875, 2.   ])
        """
        phi = np.linspace(0, 2, n)
        for p in phi:
            x = np.cos(p*np.pi)
            y = np.sin(p*np.pi)
            _phi_quadrant = cls.phi_quadrant(p)
            _xy_quadrant = cls.xy_quadrant(x,y)
            print( " p %10.4f  phi_quadrant %d   cosPhi %10.4f sinPhi %10.4f  xy_quadrant %d " % (p, _phi_quadrant, x, y, _xy_quadrant ))
        pass

    def __init__(self, phi, debug=False ):
        cosPhi, sinPhi = np.cos(phi*np.pi), np.sin(phi*np.pi)

        #n_quadrant = self.phi_quadrant(phi)
        n_quadrant = self.xy_quadrant(cosPhi, sinPhi)
        n = np.array( [  sinPhi, -cosPhi, 0. ]  )

        if debug:
            log.info( " phi %10.3f cosPhi %10.3f sinPhi %10.3f  n_quadrant %d " % (phi, cosPhi, sinPhi, n_quadrant )) 
        pass 

        self.n_quadrant = n_quadrant 
        self.n = n  
        self.debug = debug 

    def intersect(self, ray_origin, ray_direction, t_min, quadcheck=True ):
        """


        phi=1  -(*)- - - - - O-----*--+-----*------ X   phi = 0.  
                /                 /   |      \
               /                 /    |       \
              /                 /     n        \
                               /                \
                              /

        How to generalize disqualification ? 

        Check the signs of (x,y) at the intersect
        corresponds to signs of (cosPhi, sinPhi) of the plane 

        """
        n = self.n
        n_quadrant = self.n_quadrant
        debug = self.debug 

        dn = np.dot(ray_direction, n )
        on = np.dot(ray_origin,    n )
        t_cand = t_min if dn == 0. else -on/dn  
         
        xyz = ray_origin + t_cand*ray_direction 
        x = xyz[0]
        y = xyz[1]
        quadrant = self.xy_quadrant(x,y)

        if quadcheck:
            valid_intersect = t_cand > t_min and quadrant == n_quadrant  
        else:
            valid_intersect = t_cand > t_min
        pass

        if debug:
            afmt_ = lambda a:"%7.3f %7.3f %7.3f" % (a[0], a[1], a[2])
            print(" ray_origin %s ray_direction %s xyz %s t_cand %7.3f  quad %d n_quad %d valid_intersect %d" %
               (afmt_(ray_origin), afmt_(ray_direction), afmt_(xyz), t_cand, quadrant, n_quadrant, valid_intersect )) 
        pass  
        if valid_intersect: 
            isect = np.zeros(4)
            isect[:3] = n 
            isect[3] = t_cand
        else:
            isect = None
        pass
        return isect 
 

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
        t0 = t_min if dn0 == 0. else -on0/dn0  

        dn1 = np.dot(ray_direction, n1 )
        on1 = np.dot(ray_origin,    n1 )
        t1 = t_min if dn1 == 0. else -on1/dn1 

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

    def from_XY(self, n, modes=[0,1,2,3], t_min=0, shifted=True, quadcheck=True  ):
        geom = self.geom
        ipos = np.zeros( (n*len(modes), 3) ) 
        iray = np.zeros( (n*len(modes), 2, 3) ) 
        offset = 0 
        for mode in modes: 
            for i in range(n):
                j = i - n/2 
                if mode == 0:  # shoot upwards from X axis, or shifted line
                    dx = 0
                    dy = 1
                    ox = j*0.1
                    oy = -10. if shifted else 0. 
                elif mode == 1: # shoot downwards from X axis, or shifted line
                    dx = 0
                    dy = -1 
                    ox = j*0.1
                    oy = 10. if shifted else 0.  
                elif mode == 2: # shoot to right from Y axis, or shifted line 
                    dx = 1
                    dy = 0 
                    ox = -10. if shifted else 0. 
                    oy = j*0.1 
                elif mode == 3: # shoot to left from Y axis, or shifted line
                    dx = -1
                    dy = 0 
                    ox = 10. if shifted else 0.
                    oy = j*0.1 
                pass
                ray_origin    = np.array( [ox, oy, 0 ] )    
                ray_direction = np.array( [dx, dy, 0 ] )    

                iray[i+offset,0] = ray_origin
                iray[i+offset,1] = ray_direction

                isect = geom.intersect( ray_origin, ray_direction, t_min, quadcheck=quadcheck )

                if not isect is None:
                    ipos[i+offset] = isect[3]*ray_direction + ray_origin 
                pass
            pass
            offset += n 
        pass
        self.ipos = ipos
        self.iray = iray


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    HalfPlane.test_quadrant()

    #geom = HalfPlane( 0.0 )
    #geom = HalfPlane( 0.25 )
    #geom = HalfPlane( 0.50 )
    #geom = HalfPlane( 0.75 )
    #geom = HalfPlane( 1.00 )
    #geom = HalfPlane( 1.25 )
    geom = HalfPlane( 1.5, debug=True)    # problem at 1.5
    #geom = HalfPlane( 1.75, debug=False)   
    #geom = HalfPlane( 2.0, debug=False)    # problem at 2.0

    #geom = PhiCut( 0.25, 0.1 )

    #modes = [0,1,2,3]
    modes = [2,]

    sc = Scanner(geom)
    sc.from_XY(100, modes=modes, quadcheck=True )
    
    size = np.array([1280, 720])
    pl = pv.Plotter(window_size=size*2 )
    #pl.camera.ParallelProjectionOn()
    pl.view_xy() 

    mask = np.logical_and( np.abs(sc.ipos[:,1]) < 10. , np.abs(sc.ipos[:,0]) < 10. )   
    pl.add_points( sc.ipos[mask],  color="white" )  
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


