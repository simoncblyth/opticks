#!/usr/bin/env python
"""
G4OKTest.py
============

Transform global positions of photons hitting sensors
into the local frame and plot them all together.

TODO:

* show photon directions 
* plot step records
* SDF distances of hits 

"""
import os, logging, sys, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.hismask import HisMask
from opticks.ana.OpticksIdentity import OpticksIdentity
from opticks.ana.ggeo import GGeo
hismask = HisMask()

try:
    import pyvista as pv   
except ImportError:
    pv = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
        log.info("chdir %s " % os.getcwd())
    pass
    np.set_printoptions(suppress=True, linewidth=200)

    gg = GGeo()

    ## hmm need a standard place for detector level stuff like this 
    sentid = np.load(os.path.expandvars("$TMP/G4OKTest/sensorData.npy")).view(np.uint32)[:,3]  

    #name = "ht"
    name = "ox" 

    ox = np.load("%s.npy" % name)
    ox_land = ox[ox.view(np.int32)[:,3,1] != -1]   # photons that land on sensors, only some become hits  

    lpos = np.zeros( (len(ox_land),4), dtype=np.float32 )
    ldir = np.zeros( (len(ox_land),4), dtype=np.float32 )
    lpol = np.zeros( (len(ox_land),4), dtype=np.float32 )

    for i,oxr in enumerate(ox_land):
        oxf = oxr[3].view(np.int32)
        bnd,sidx,idx,pflg = oxf  

        tid = sentid[sidx]  # sensor index to triplet id
        ridx,pidx,oidx = OpticksIdentity.Decode(tid)   # triplet from id
        tr = gg.get_transform(ridx,pidx,oidx)
        it = gg.get_inverse_transform(ridx,pidx,oidx)
        #it2 = np.linalg.inv(tr)   ## better to do this once within ggeo 
        #assert np.allclose( it, it2 )

        gpos = oxr[0]
        gdir = oxr[1]
        gpol = oxr[2]

        gpos[3] = 1.  # replace time with 1. for transforming a position
        gdir[3] = 0.  # replace weight with 0. for transforming a direction 
        gpol[3] = 0.  # replace wavelength with 0. for transforming a direction 

        lpos[i] = np.dot( gpos, it )
        ldir[i] = np.dot( gdir, it )
        lpol[i] = np.dot( gpol, it )

        print("bnd/sidx/idx/pflg  %20s %15s  tid %8x  (ridx/pidx/oidx %d %4d %4d)  %s " % (oxf, hismask.label(pflg), tid,ridx,pidx,oidx, lpos[i] ))
        #print(tr)
    pass

 
    if pv: 
        pl = pv.Plotter(window_size=2*np.array([1024,768], dtype=np.int32))
        pl.add_points(lpos[:,:3])
        pl.show_grid()
        log.info("Showing the VTK/pyvista plotter window, it may be hidden behind other windows. Enter q to quit.")
        cpos = pl.show()
        log.info(cpos)



 

