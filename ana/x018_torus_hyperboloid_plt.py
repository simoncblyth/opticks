#!/usr/bin/env python
"""
Continuing from tboolean-12

See also

x018.py x019.py x020.py x021.py 
    these were manually created from the generated g4code 
    but could be generated too

shape.py
    rationalizations replacing tubs-torus with cons

notes/issues/torus_replacement_on_the_fly.rst


// arghh : G4Hype cannot be z-cut : so would have to intersect with cylinder 
// this makes me want to use a cone for the neck instead 

"""

import logging
log = logging.getLogger(__name__)
import numpy as np, math 


import matplotlib.pyplot as plt
from opticks.ana.torus_hyperboloid import Tor, Hyp

from opticks.ana.shape import Shape, ellipse_closest_approach_to_point

class Ellipsoid(Shape):pass
class Tubs(Shape):pass
class Torus(Shape):pass
class Cons(Shape):pass
class Hype(Shape):pass
class UnionSolid(Shape):pass
class SubtractionSolid(Shape):pass
class IntersectionSolid(Shape):pass



class X018(object):
    """
    AIMING TO PHASE THIS OUT, use instead x018.py 

    G4VSolid* make_solid()
    { 
        G4ThreeVector A(0.000000,0.000000,-23.772510);
        G4ThreeVector B(0.000000,0.000000,-195.227490);
        G4ThreeVector C(0.000000,0.000000,-276.500000);
        G4ThreeVector D(0.000000,0.000000,92.000000);


        G4VSolid* d = new G4Ellipsoid("PMT_20inch_inner_solid_1_Ellipsoid0x4c91130", 249.000000, 249.000000, 179.000000, -179.000000, 179.000000) ; // 3
        G4VSolid* g = new G4Tubs("PMT_20inch_inner_solid_2_Tube0x4c91210", 0.000000, 75.951247, 23.782510, 0.000000, CLHEP::twopi) ; // 4
        G4VSolid* i = new G4Torus("PMT_20inch_inner_solid_2_Torus0x4c91340", 0.000000, 52.010000, 97.000000, -0.000175, CLHEP::twopi) ; // 4
        
        G4VSolid* f = new G4SubtractionSolid("PMT_20inch_inner_solid_part20x4cb2d80", g, i, NULL, A) ; // 3
        G4VSolid* c = new G4UnionSolid("PMT_20inch_inner_solid_1_20x4cb30f0", d, f, NULL, B) ; // 2

        G4VSolid* k = new G4Tubs("PMT_20inch_inner_solid_3_EndTube0x4cb2fc0", 0.000000, 45.010000, 57.510000, 0.000000, CLHEP::twopi) ; // 2
        
        G4VSolid* b = new G4UnionSolid("PMT_20inch_inner_solid0x4cb32e0", c, k, NULL, C) ; // 1
        G4VSolid* m = new G4Tubs("Inner_Separator0x4cb3530", 0.000000, 254.000000, 92.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4VSolid* a = new G4IntersectionSolid("PMT_20inch_inner1_solid0x4cb3610", b, m, NULL, D) ; // 0
        return a ; 
    } 


                           a

                      b             m(D)
                 
                c          k(C)

            d     f(B) 

                g(B)  i(B+A)

    """
    def __init__(self, mode=0):
        A = np.array( [0, -23.772510] )
        C = np.array( [0, -276.500000] )
        D = np.array( [0, 92.000000] )

        B0 = np.array( [0, -195.227490] )
        z0 = B0[1] + A[1]    ## offset of torus (ellipsoid frame)

        print("z0 %s offset of torus (ellipsoid frame)" % z0 )

        self.z0 = z0

        i = Torus("i", [ 52.010000, 97.000000] )   
        # 97-51.01 = 44.99 ( slightly inside the tube radius 45.01 ? accident or CSG planning ?)
        r = i.param[0] 
        R = i.param[1]

        self.r = r
        self.R = R

        d = Ellipsoid("d", [249.000, 179.000 ] )
        ex = d.param[0]
        ez = d.param[1]

        self.ex = ex
        self.ez = ez

        k = Tubs("k", [45.010000, 57.510000] )
        bx = k.param[0]
        self.bx = bx


        p = ellipse_closest_approach_to_point( ex, ez, [R,z0] ) # center of torus RHS circle 
        print(" p %s " % repr(p) )

        pr = p[0]
        pz = p[1]    # z of ellipse to torus-circle closest approach point (ellipsoid frame) 

        self.p = p 


        if mode == 0:
            g = Tubs("g", [75.951247,23.782510] )
            f = SubtractionSolid("f", [g,i,A] )
            B = B0
        elif mode == 1:
            r2 = 83.9935              # at torus/ellipse closest approach point : huh same as pr no (maybe not quite)
            r1 = R - r               

            mz = (z0 + pz)/2.         # mid-z cone coordinate (ellipsoid frame) 
            hz = (pz - z0)/2.         # cons half height  

            f = Cons("f", [r1,r2,hz ] )
            B = np.array( [0, mz] )

            print(" replacment Cons %s offset %s " % (repr(f),repr(B)))

        elif mode == 2:

            r0 = R - r
            rw,zw = pr,pz-z0   # target the closest approach point  

            zf = Hyp.ZF( r0, zw, rw )
            hyp = Hyp( r0, zf )
         
            stereo = hyp.stereo
            hhh = pz - z0    # hype-half-height 

            f = Hype("f", [r0,stereo,hhh] ) 
            B = np.array( [0, z0] )

        else:
            assert 0 
        pass

        c = UnionSolid("c", [d,f,B] )

        m = Tubs("m", [254.000000, 92.000000] )
        b = UnionSolid("b", [c,k,C] )
        a = IntersectionSolid("a", [b,m,D] )

        self.root = a 
        self.sz = 400

        self.ellipse = d
        self.torus = i
        #self.cyneck = g
        self.f = f 
    pass



if __name__ == '__main__':

    #x = X018(mode=0)  # crazy cylinder-torus neck
    x = X018(mode=1)  # cone neck 
    #x = X018(mode=2)  # hype neck 

    print "x.f ", x.f


    sz = x.sz

    r = x.r
    R = x.R
    z0 = x.z0   # natural torus frame offset in ellipse frame

    


    #ch = x.cyneck.param[1]*2   # full-height of cylinder neck 
    #cr = x.cyneck.param[0]     # half-width of cylinder neck, ie the radius 

    ex = x.ex
    ez = x.ez

    p = x.p  # closest approach of RHS torus circle center to a point on the ellipse  


    tor = Tor(R,r)
    assert tor.rz(0) == R - r 
    assert tor.rz(r) == R  
    assert np.all(tor.rz([0,r]) == np.asarray( [R-r, R] ) )


    tz0 = 0 
    tz1 = r      # how far in z to draw the torus and hype parametric lines
    #tz1 = ch 

    tz = np.linspace( tz0, tz1, 100)
    tr = tor.rz(tz)


    r0 = R - r     # Hyperboloid waist radius at z=0 (natural frame)
    # defining parameter of Hyperboloid by targetting an rz-point (natural frame)
    
    #rw,zw = R, r   # target the top of the torus : gives wider neck than  cy-tor 
    rw,zw = p[0],p[1]-z0   # target the closest approach point  
    halfZlen = p[1]-z0

    zf = Hyp.ZF( r0, zw, rw )
    hyp = Hyp( r0, zf )
    print hyp
    print "hyp halfZLen ", halfZlen 

    hr = hyp.rz(tz)

    tz += z0    


    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("x018_torus_hyperboloid_plt")

    ax = fig.add_subplot(111)

    zoom = False
    #zoom = True

    if zoom:
        zmp = np.array( [R, z0, 1.5*r, 1.5*r] )
        ax.set_xlim([zmp[0]-zmp[2],zmp[0]+zmp[2]])
        ax.set_ylim([zmp[1]-zmp[3],zmp[1]+zmp[3]])
    else:
        ax.set_ylim([-350,200])
        ax.set_xlim([-300,300])
    pass

    for pt in x.root.patches():
        print "pt ", pt
        ax.add_patch(pt)
    pass

    #rhs = Circle( [R,z0], radius=r )
    #ax.add_patch(rhs)
    #ell = Ellipse( [0,0], 2*ex, 2*ez ) 
    #ax.add_patch(ell)

    #ax.scatter( e[:,0], e[:,1], s=1 ) 

    ax.scatter( p[0], p[1] , marker="*")
    ax.scatter( -p[0], p[1] , marker="*" )

    ax.plot( [R, p[0]], [z0, p[1]] )
    ax.plot( [-R, -p[0]], [z0, p[1]] )

    ax.plot( [R-r, p[0]], [z0, p[1]] )
    ax.plot( [-R+r, -p[0]], [z0, p[1]] )

    ax.plot(  tr, tz )
    ax.plot( -tr, tz )

    ax.plot(  hr, tz , linestyle="dashed")
    ax.plot( -hr, tz , linestyle="dashed")



   
    fig.show()


