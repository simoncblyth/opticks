#!/usr/bin/env python
"""
cross2D_angle_range_without_trig.py 
=====================================

Working out how to use 2D cross product to select 
a range of phi directions without using arc-trig funcs 
like atan2. 

"""


def cross2D( a , b ):
    """
    https://www.nagwa.com/en/explainers/175169159270/


    2D cross product definition::

         A = Ax i + Ay j 

         B = Bx i + By j


         A ^ B =  (Ax By - Bx Ay ) k 

                = |A||B| sin(AB_angle) k

                = sin(AB_angle) k         when A and B are normalized 


    Anti-commutative::

          A ^ B = - B ^ A 

    """
    assert a.shape[-1] == 3
    assert b.shape[-1] == 3

    if a.ndim == 1 and b.ndim == 1:
        c = a[0]*b[1] - b[0]*a[1]
    elif a.ndim == 1 and b.ndim == 2:
        c = a[0]*b[:,1] - b[:,0]*a[1]
    elif a.ndim == 2 and b.ndim == 1:
        c = a[:,0]*b[1] - b[0]*a[:,1]
    else:
        c = None
    pass
    assert not c is None
    print("a.ndim : %d a.shape %s " % (a.ndim, str(a.shape)))
    print("b.ndim : %d b.shape %s " % (b.ndim, str(b.shape)))
    print("c.ndim : %d c.shape %s " % (c.ndim, str(c.shape)))
    return c 





import numpy as np
import pyvista as pv

def compose_XY( pl ):
     """



               Y
               | 
               U 
               | 
               | 
               |
               L---------- X
              /
             /
            E 
           /
          Z

     """
     look = np.array( [0,0,0], dtype=np.float32 )
     up = np.array( [0,1,0], dtype=np.float32 )
     eye = np.array( [0,0,10], dtype=np.float32 )
     zoom = 1 

     pl.set_focus(    look )
     pl.set_viewup(   up )
     pl.set_position( eye, reset=False )  
     pl.camera.Zoom(zoom)
     pl.show_grid()

if __name__ == '__main__':


     SIZE = np.array([1280, 720])
     pl = pv.Plotter(window_size=SIZE*2 )

     #phiMode = "pacman"
     #phiMode = "pacman_small"
     #phiMode = "ppquad"
     phiMode = "nnquad"
     #phiMode = "uphemi"

     if phiMode == "pacman": 
         phiStart = 0.25  
         phiDelta = 1.50    
     elif phiMode == "pacman_small": 
         phiStart = 0.10
         phiDelta = 1.80    
     elif phiMode == "ppquad":
         phiStart = 0.00
         phiDelta = 0.25
     elif phiMode == "uphemi":
         phiStart = 0.00
         phiDelta = 1.00
     elif phiMode == "nnquad":
         phiStart = 1.00
         phiDelta = 0.5
     else:
         assert 0 
     pass

     phi0_pi = phiStart
     phi1_pi = phiStart+phiDelta 
     phi_pi = np.linspace( phi0_pi,  phi1_pi, 100 )
     phi = np.pi*phi_pi
     cosPhi = np.cos( phi ) 
     sinPhi = np.sin( phi ) 

 
     ## edge vectors : as standin for geometry 
     evec0 = np.array( [cosPhi[0], sinPhi[0], 0], dtype=np.float32 )
     evec1 = np.array( [cosPhi[-1], sinPhi[-1], 0], dtype=np.float32 )

     ## edge lines from origin 
     ll = np.zeros( (2, 2, 3), dtype=np.float32 )
     ll[:,0] = (0,0,0)
     ll[0,1] = evec0
     ll[1,1] = evec1

     for i in range(len(ll)):
         pl.add_lines( ll[i].reshape(-1,3), color="blue" )
     pass  
     
     ## around the circular arc points 
     pos = np.zeros( (len(phi), 3 ), dtype=np.float32 )
     pos[:,0] = cosPhi 
     pos[:,1] = sinPhi
     pos[:,2] = 0.
     pl.add_points( pos, color="white" ) 

   

     ## random phi in full range 
     rnum = 100  
     #rphi = 2.*np.random.random_sample(rnum) 
     rphi = np.linspace( 0., 2., rnum)

     rpos = np.zeros( (rnum, 3), dtype=np.float32 )
     rpos[:,0] = np.cos(np.pi*rphi)
     rpos[:,1] = np.sin(np.pi*rphi)
     rpos[:,2] = 0

     s01 = cross2D(evec0, evec1)  # sine of angle evec0 -> evec1 
     s0 = cross2D(evec0, rpos )   # sine of angle evec0 -> rpos
     s1 = cross2D(evec1, rpos )   # sine of angle evec1 -> rpos 

     ## hmm could flip the s1 definition 
     ## to make the rpos inbetween case have both signs the same      


     # check anti-commutativity 
     s0_check = -cross2D(rpos, evec0) ; assert np.all( s0 == s0_check )  
     s1_check = -cross2D(rpos, evec1) ; assert np.all( s1 == s1_check )  


     """
     cross2D(evec0, evec1) > 0   means angle less than 1. (pi)

     The sign-of-the-cross2D-sine tells you which "side of the line", 
     so the *logical_and* of two such signs gives the region between the lines.
     This works fine when the angle between evec0 and evec1 is less than pi, 
     selecting less than half the cake.  
     

               
                Y
                .        /  s0 > 0
                .       / + + + + + + + 
                .      / + + + + + + + 
                .    evec0 + + + + + + 
                .    / + + + + + + +
                .   / + + + + + + + 
                .  / + + + + + + +      
                . / + + + + + + +           
                ./ + + + + + + + +          
                O . . . . . . . . . . . . . . . X
               .| + + + + + + + +  
              . | + + + + + + + +           
             .  | + + + + + + + +     
            .   | + + + + + + + +         
           .    | + + + + + + + +    
          .     | + + + + + + + + 
         .     evec1  + + + + + + 
                | + + + + + + + + 
                |   s1 < 0 


      When the angle between evec0 and evec1 is more than pi 
      need to use *logical_or* of between the two signs 
      to be more inclusive and thus greedily select more than half of the cake. 

       evec0
        \ + +   .
         \ + +  .
          \ + + .
           \ + +. 
            \ + .
             \ +.+
              \ . +
               \. + 
                O + +    
                |. + +
                | . + +
                |+ . + +
                |+ +. + +
                |+ + . + +
                |+ + +. + +
                |+ + + . + +
             evec1
 
     """

     rr = np.zeros( (rnum,2,3), dtype=np.float32 )
     rr[:,0] = (0,0,0)
     rr[:,1] = rpos

     #mask = np.logical_and( s0 > 0. , s1 > 0. )    # +Y quad
     #mask = np.logical_or(  s0 > 0. , s1 > 0. )     # not -Y quad
     #mask = np.logical_xor(  s0 > 0. , s1 > 0. )     # +X and -X

     #mask = np.logical_and( s0 < 0. , s1 < 0. )    # -Y quad
     #mask = np.logical_or( s0 < 0. , s1 < 0. )      # not +Y quad 
     #mask = np.logical_xor( s0 < 0. , s1 < 0. )      # +X and -X 

     #mask = np.logical_and( s0 < 0. , s1 > 0. )    # +X quad 
     #mask = np.logical_or( s0 < 0. , s1 > 0. )      # not -X quad 
     #mask = np.logical_xor( s0 < 0. , s1 > 0. )      # +Y and -Y 

     #mask = np.logical_and( s0 > 0. , s1 < 0. )     # -X quad           ## one to use for ppquad
     #mask = np.logical_or(  s0 > 0. , s1 < 0. )      # not +X quad    ## one to use for pacman, pacman_small 
     #mask = np.logical_xor(  s0 > 0. , s1 < 0. )      # +Y and -Y

     if s01 > 0.:   # angle less than pi 
         mask = np.logical_and( s0 > 0. , s1 < 0. )     
     else:          # angle greater than pi 
         mask = np.logical_or( s0 > 0. , s1 < 0. )     
     pass

     urr = rr[mask]
     
     for i in range(len(urr)):
         pl.add_lines( urr[i].reshape(-1,3), color="green" )
     pass  
 

     compose_XY(pl)
     cp = pl.show()


