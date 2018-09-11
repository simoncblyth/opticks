#!/usr/bin/env python

import numpy as np

class F4(object):
    def __init__(self, x,y,z,w):
        self.x = x 
        self.y = y 
        self.z = z 
        self.w = w 


dom = F4(60., 820., 0., 0.)
hc = 1240.*1e-6   # MeV.nm


def lerp(a, b, t ):
    return (1.-t)*a + t*b 

def boundary_sample_reciprocal_domain(u, flip=False):
    boundary_domain_reciprocal = F4(1./dom.x, 1./dom.y, 0,0 )
    if flip:
        iw = lerp( boundary_domain_reciprocal.y, boundary_domain_reciprocal.x,  u ) 
    else:
        iw = lerp( boundary_domain_reciprocal.x, boundary_domain_reciprocal.y,  u ) 
    pass
    return 1./iw 


def boundary_sample_reciprocal_domain_v3( u ):
    """
    ::
                  1 - u          u         (flipped)
          iw  =  --------  +   -----    
                    b            a

           
       a.b.iw =   a( 1 -u ) +  b u 


                        a.b
           w  =   ------------------   
                    a(1-u) + b u      

    """
    a = dom.x
    b = dom.y 
    return  a*b/lerp( a, b, u )


def sampledEnergy(u):
    """
    In [12]: (1240./1e6)/2.0664e-05
    Out[12]: 60.0077429345722

    In [13]: (1240./1e6)/1.512e-06
    Out[13]: 820.10582010582
    """

    Pmin = hc/dom.y
    Pmax = hc/dom.x
    return lerp( Pmin, Pmax, u )



if __name__ == '__main__':

    uu = [0,0.740219,1] 


    w = np.zeros( (len(uu), 4), dtype=np.float64) 

    for i, u in enumerate(uu):
        w[i,0]=boundary_sample_reciprocal_domain(u, flip=False) 
        w[i,1]=boundary_sample_reciprocal_domain(u, flip=True) 
        en = sampledEnergy(u) 
        w[i,2] = hc/en 
        w[i,3]=boundary_sample_reciprocal_domain_v3(u) 

        print "%10.4f : %10.4f : %10.4f : %10.4f : %10.4f : %10.4f " % (u, w[i,0], w[i,1], w[i,2], w[i,3], en  )
    pass










