#!/usr/bin/env python





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

    for u in [0,0.740219,1]:
        w0=boundary_sample_reciprocal_domain(u, flip=False) 
        w1=boundary_sample_reciprocal_domain(u, flip=True) 
        en = sampledEnergy(u) 
        w2 = hc/en 
        print "%10.4f : %10.4f : %10.4f : %10.4f : %10.4f " % (u, w0, w1, w2, en  )
    pass










