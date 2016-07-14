#!/usr/bin/env python
"""
xrainbow.py : Rainbow Expectations
====================================

Using derivations from: Jearl D. Walker
"Multiple rainbows from single drops of water and other liquids",  

* http://www.patarnott.com/atms749/pdf/MultipleRainbowsSingleDrops.pdf

Alexanders dark band, between the 1st and 2nd bows 
(due to no rays below min deviation for each bow)


See Also
----------

Abandoned approach to analytic rainbows in env/opticksnpy/rainbow*.py 


Polarization Check
-------------------

Plane of incidence defined by initial direction vector 
(a constant vector) and the surface normal at point of incidence, 
which will be different for every intersection point. 

Thus need to specially prepare the polarizations in order to
arrange S-polarized incident light. Basically need to 
calculate surface normal for all points of sphere.

S-polarized :  perpendicular to the plane of incidence


Rendering Spectra
-------------------

Comparison of several approaches to handling out of gamut spectral colors

* http://www-rohan.sdsu.edu/~aty/explain/optics/rendering.html

Collection of color science links

* http://www.midnightkite.com/color.html


Rainbow Calculations
----------------------

The mathematical physics of rainbows and glories, John A. Adam

* http://ww2.odu.edu/~jadam/docs/rainbow_glory_review.pdf



Optics of a water drop
------------------------

* http://www.philiplaven.com/index1.html

Fig 4, Provides relative intensities of S/P-polarizations 
at each step for primary bow.  


Thru multiple Relect/Transmit : is S/P polarization retained ?
------------------------------------------------------------------

S/P polarization is defined with respect to the surface normal 
at the point if incidence.  

Every reflection/transmission happens in the same plane, so that 
suggests  



Maybe the assumption of constant polarization
state is in fact a valid one ?  

* does this depend on the geometry 







"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from opticks.ana.base import opticks_environment
from opticks.ana.droplet  import Droplet


class XRainbow(object):
    def __init__(self, w, boundary, k=1 ):
        """
        :param w: wavelength array
        :param boundary: instance
        :param k: 1.. rainbow index, -1 direct reflection 

        Attributes:

        i 
            incident angle of minimum deviation 
        d 
            total deviation angle at minimum deviation
        n 
            refractive indices corresponding to wavelength array


        There is symmetry about the ray at normal incidence so consider
        a half illuminated drop.

        Deviation angles are calculated in range 0:360 degrees 

           k    red    blue
           1   137.63  139.35
           2   230.37  233.48


        0:180 
             signifies rays exiting in hemisphere opposite 
             to the incident hemisphere

        180:360 
             signifies rays exiting in same hemisphere  
             as incidence


        ::

            In [8]: xbow.dv
            Out[8]: 
            array([ 2.553,  2.553,  2.553,  2.553,  2.553,  2.553,  2.553,  2.553,
                    2.513,  2.488,  2.47 ,  2.456,  2.447,  2.441,  2.435,  2.43 ,
                    2.426,  2.422,  2.42 ,  2.418,  2.416,  2.414,  2.412,  2.41 ,
                    2.408,  2.407,  2.407,  2.405,  2.405,  2.403,  2.402,  2.402,
                    2.402,  2.4  ,  2.4  ,  2.4  ,  2.399,  2.397,  2.397])

            In [9]: xbow.w
            Out[9]: 
            array([  60.   ,   79.737,   99.474,  119.211,  138.947,  158.684,
                    178.421,  198.158,  217.895,  237.632,  257.368,  277.105,
                    296.842,  316.579,  336.316,  356.053,  375.789,  395.526,
                    415.263,  435.   ,  454.737,  474.474,  494.211,  513.947,
                    533.684,  553.421,  573.158,  592.895,  612.632,  632.368,
                    652.105,  671.842,  691.579,  711.316,  731.053,  750.789,
                    770.526,  790.263,  810.   ])

 
        """
        self.boundary = boundary
        self.droplet = Droplet(boundary)
        self.w = w  
        self.k = k

        redblue = np.array([780., 380.])
        self.dvr = self.droplet.deviation_angle(redblue, k)
        self.dv = self.droplet.deviation_angle(w, k)


    def dbins(self, nb, window=[-0.5,0.5]):
        """
        :param nb: number of bins
        :param window: degress of window around predicted min/max deviation
        """
        d = self.dvr 
        dmin = d.min() + window[0]*deg
        dmax = d.max() + window[1]*deg
        return np.linspace(dmin,dmax, nb)


    def refractive_index(self): 
        """
        Plateau in refractive index below 330nm for Glass, 
        edge of data artifact

        ::

            xbow.refractive_index()
            plt.show()

        """
        wd = np.arange(80,820,10)
        nd = self.boundary.imat.refractive_index(wd)  

        plt.plot(wd, nd)

        return wd, nd



class XFrac(object):
    """
    S-pol/P-pol (polarized perperndicular/parallel to plane of incidence) intensity fraction
    """
    def __init__(self, n, k=np.arange(1,6)):

        i = np.arccos( np.sqrt((n*n - 1.)/(k*(k+2.)) ))  # bow angle
        r = np.arcsin( np.sin(i)/n )                    

        # rainbow paper 
        #     Jearl D. Walker p426, demo that ek1 is indep of n 
        #
        #     derivations assume that the S/P polarization will stay the same 
        #     across all the reflections, that seems surprising 
        # 
        #     swapped the sin and tan for S/P factors
        # 

    
        # perpendicular to plane of incidence, S-pol 
        fs = np.power( np.sin(i-r)/np.sin(i+r) , 2 )
        ts = 1 - fs 
        s = ts*ts*np.power(fs, k)

        # parallel to plane of incidence, P-pol 
        fp = np.power( np.tan(i-r)/np.tan(i+r) , 2 )
        tp = 1 - fp
        p = tp*tp*np.power(fp, k)   


        kk = np.sqrt( k*k + k + 1 )
        qq = (kk - 1)/(kk + 1)
        pq = np.power((1-qq*qq),2)*np.power(qq, 2*k)      

        self.i = i
        self.r = r

        self.fp = fp
        self.tp = tp
        self.p = p

        self.fs = fs
        self.ts = ts
        self.s = s

        self.t = s + p 


        self.kk = kk 
        self.qq = qq 
        self.pq = pq










if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    from opticks.ana.boundary import Boundary   

    boundary = Boundary("Vacuum///MainH2OHale") 

    w = np.linspace(60.,810., 39)

    k = 1  

    xbow = XRainbow(w, boundary, k=k )


 
