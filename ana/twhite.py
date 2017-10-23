#!/usr/bin/env python
"""
twhite.py: Wavelength Distribution Check
============================================

Creates plot comparing simulated photon wavelength spectrum 
from :doc:`../tests/twhite` against blackbody expectation.

This is checking the *source_lookup* implementation and 
the inverse CDF *source_texture* that it uses.  

.. code-block:: cpp

    ## optixrap/cu/wavelength_lookup.h

    014 rtTextureSampler<float, 2>  source_texture ;
     15 rtDeclareVariable(float4, source_domain, , );
     ..
     41 static __device__ __inline__ float source_lookup(float u)
     42 {
     43     float ui = u/source_domain.z + 0.5f ;
     44     return tex2D(source_texture, ui, 0.5f );  // line 0
     45 }
     46 
     47 static __device__ __inline__ void source_check()
     48 {
     49     float nm_a = source_lookup(0.0f);
     50     float nm_b = source_lookup(0.5f);
     51     float nm_c = source_lookup(1.0f);
     52     rtPrintf("source_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
     53 }

    ## optixrap/cu/torchstep.h

    241 __device__ void
    242 generate_torch_photon(Photon& p, TorchStep& ts, curandState &rng)
    243 {
    244       p.wavelength = ts.wavelength > 50.f ? ts.wavelength : source_lookup(curand_uniform(&rng));  // Planck black body source 6500K standard illuminant 
    245 


See Also
----------

* :doc:`source_debug`


"""

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from opticks.ana.base import opticks_main
from opticks.ana.evt import Evt
from opticks.ana.planck import planck



if __name__ == '__main__':
    plt.ion()
    args = opticks_main(tag="1", det="white", src="torch")

    ## tag = "1"   ## dont have any tag 1 anymore 
    ## tag = "15"     ## so added tag 15,16 to ggv-rainbow with wavelength=0 which is default black body 

    try:
        evt = Evt(tag=args.tag, det=args.det, src=args.src, args=args )
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)


    if not evt.valid:
       log.fatal("failed to load evt %s " % repr(args))
       sys.exit(1) 


    wl = evt.wl
    w0 = evt.recwavelength(0)  

    w = wl
    #w = w0

    wd = np.linspace(60,820,256) - 1.  
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


