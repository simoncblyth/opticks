#!/usr/bin/env python
"""
Wavelength Distribution Debugging
====================================

Compare simulated photon wavelengths against blackbody expectation


w0 sel.recwavelength(0) Features
----------------------------------

Without selection sel.recwavelength(0) from ggv-newton:

* length of 500000

* three bin spike at lower bound around 60nm, comprising about 7000 photons
  (not present in the uncompressed wp)

* plateau from 60~190 nm

* normal service resumes above 190nm with good
  match to Planck black body curve

* 256 unique linspaced values, a result of the compression:: 

    In [36]: np.allclose(np.linspace(60,810,256),np.unique(w))
    Out[36]: True

::

In [45]: count_unique(w0)
Out[45]: 
array([[   60.   ,  1605.   ],
       [   62.941,  1676.   ],
       [   65.882,  3420.   ],
       [   68.824,    49.   ],
       [   71.765,    59.   ],
       [   74.706,    46.   ],
       [   77.647,    47.   ],
       ...
       [  174.706,    58.   ],
       [  177.647,    46.   ],
       [  180.588,    57.   ],
       [  183.529,    48.   ],
       [  186.471,   238.   ],
       [  189.412,   324.   ],
       [  192.353,   383.   ],


wp evt.wavelength features
----------------------------

* no 80nm spike 3 bin spike

* same plateau as w0


comparing w0 and wp
---------------------

* 6606 out of .5M discrepant 

::

    In [64]: b = w0 - wp <  -100 

    In [65]: w0[b]
    Out[65]: array([ 62.941,  65.882,  60.   , ...,  62.941,  60.   ,  65.882], dtype=float32)

    In [66]: wp[b]
    Out[66]: array([ 815.307,  820.   ,  813.841, ...,  815.74 ,  812.291,  820.   ], dtype=float32)

Plotting the uncompressed wavelength of the low peak, they are all > 812 nm with spike at 820nm
Hmm theres a discrepancy of 10 nm in domain maximum, the compression range tops out at 810 nm, so 
wavelengths beyond there appear to be loosing some bits and ending up at low end.::


    In [75]: w0.max()   
    Out[75]: 810.0

    In [76]: wp.max()
    Out[76]: 820.0


::

    In [2]: np.load("OPropagatorF.npy")
    Out[2]: 
    array([[[   0.,    0.,    0.,  700.]],

           [[   0.,    7.,    7.,    0.]],

           [[  60.,  810.,   20.,  750.]]], dtype=float32)


        255.*(60. - 60.)/750. ->  255.*0  -> 0 

        255.*(810.-60.)/750.  -> 255.*1.0 -> 255.

        255.*(820.-60.)/750.  -> 255.*1.0133 -> 258.4



    143     float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 
    144 
    145     qquad qpolw ;
    146     qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
    147     qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    148     qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    149     qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;


     39 static __device__ __inline__ float source_lookup(float u)
     40 {     
     41     float ui = u/source_domain.z + 0.5f ;
     42     return tex2D(source_texture, ui, 0.5f );  // line 0 
     43 }         

           --> ui = u*float(nx) + 0.5      (nx is 256 coming from GSourceLib::icdf_length = 256)
                     
                  u = 0 ->   0.5
                  u = 1 ->   256 + 0.5       

     32     float step = 1.f/float(nx) ;
     33     optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
     34     optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny);
     35 
     36     m_context["source_texture"]->setTextureSampler(tex);
     37     m_context["source_domain"]->setFloat(domain);



Dumping from bounds see that u=1 is providing the > 810 nm wavelenths...

     source_check nm_a     60.000    506.041    820.000  
     source_check nm_a     60.000    506.041    820.000  


     25 float        GPropertyLib::DOMAIN_LOW  = 60.f ;
     26 float        GPropertyLib::DOMAIN_HIGH = 810.f ;
     27 float        GPropertyLib::DOMAIN_STEP = 20.f ;
     28 unsigned int GPropertyLib::DOMAIN_LENGTH = 39  ;
     29     
     30 unsigned int GPropertyLib::UNSET = UINT_MAX ;
     31 unsigned int GPropertyLib::NUM_QUAD = 4  ;
     32 unsigned int GPropertyLib::NUM_PROP = 4  ;
     33 
     34 GDomain<float>* GPropertyLib::getDefaultDomain()
     35 {
     36    return new GDomain<float>(DOMAIN_LOW, DOMAIN_HIGH, DOMAIN_STEP );
     37 }


Looking back at the Planck source see that are going 10nm too far in wavelength::

    simon:ggeo blyth$ ggv --gsrclib
    [2015-Dec-02 17:10:48.224924]:info: GAry::save 1d array of length 500000 to : /tmp/blackbody.npy
    simon:ggeo blyth$ 

The funny plateau is also apparent at this level::

    In [95]: bb = np.load("/tmp/blackbody.npy")

    In [96]: bb
    Out[96]: array([ 676.328,  329.759,  740.522, ...,  528.807,  413.573,  614.098], dtype=float32)

    In [97]: bb.min()
    Out[97]: 60.160744

    In [98]: bb.max()
    Out[98]: 819.99951




"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection
from env.graphics.ciexyz.planck import planck

np.set_printoptions(suppress=True, precision=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.ion()

    evt = Evt(tag="1", det="prism")

    sel = Selection(evt)

    wp = evt.wavelength
    w0 = sel.recwavelength(0)  

    w = w0


    wd = np.linspace(60,810,256) - 1.  
    # reduce bin edges by 1nm to avoid aliasing artifact in the histogram

    mid = (wd[:-1]+wd[1:])/2.     # bin middle

    pl = planck(mid, 6500.)
    pl /= pl.sum()

    counts, edges = np.histogram(w, bins=wd )
    fcounts = counts.astype(np.float32)
    fcounts  /= fcounts.sum()


    plt.close()

    plt.plot( edges[:-1], fcounts, drawstyle="steps-mid")

    plt.plot( mid,  pl ) 
    
    plt.axis( [w.min() - 100, w.max() + 100, 0, fcounts.max()*1.1 ]) 


    #plt.hist(w, bins=256)   # 256 is number of unique wavelengths (from record compression)



