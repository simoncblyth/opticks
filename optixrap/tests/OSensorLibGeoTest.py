#!/usr/bin/env python
"""

::

   ipython -i OSensorLibGeoTest.py 
   open $TMP/optixrap/tests/OSensorLibGeoTest/pixels.ppm 

Similar::

   ~/opticks/examples/UseOptiXGeometryInstancedOCtx/intersect_sdf_test.py
   ~/opticks/optickscore/IntersectSDF.cc

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

path_ = lambda stem:os.path.expandvars("/tmp/$USER/opticks/optixrap/tests/OSensorLibGeoTest/%s.npy" % stem)

## decode the packing from okc/SphereOfTransforms
itheta_ = lambda tr_identity:( tr_identity & 0x000000ff ) >> 0 
iphi_   = lambda tr_identity:( tr_identity & 0x0000ff00 ) >> 8 
index_  = lambda tr_identity:( tr_identity & 0xffff0000 ) >> 16   

class OSensorLibGeoTest(object):
    def __init__(self):
        tr = np.load(path_("transforms"))
        posi = np.load(path_("posi"))
        px = np.load(path_("pixels"))

        self.setTransforms(tr)
        self.setPosi(posi)

    def setTransforms(self, tr):
        log.info("tr {tr.shape!r}".format(tr=tr))

        tr_identity = tr[:,0,3].view(np.uint32).copy()    

        tr[:,0,3] = 0.          ## scrubbing transform identity needed before invert
        it = np.linalg.inv(tr)  ## invert all the transforms at once 

        itheta   = itheta_(tr_identity)   ## decode the packing from okc/SphereOfTransforms
        iphi     = iphi_(tr_identity)
        index    = index_(tr_identity)
        index_expected = np.arange(len(index),dtype=np.uint32) + 1  ## 1-based index
        assert np.all(index == index_expected)  
   
        log.info(" itheta.min,max %d %d " % (itheta.min(), itheta.max()))
        log.info(" iphi.min,max   %d %d " % (iphi.min(), iphi.max()))
        log.info(" index.min,max  %d %d " % (index.min(), index.max()))
 
        self.tr = tr
        self.it = it
        self.itheta = itheta 
        self.iphi = iphi 
        self.index = index

    def setPosi(self, posi):
        """
        :param posi: array of shape (height, width, 4) carrying intersect global position and geometry identity 
        """ 
        log.info("posi {posi.shape!r}".format(posi=posi))
        identity = posi[:,:,3].view(np.uint32)
        assert len(identity.shape) == 2 
        num_pix = identity.shape[0]*identity.shape[1]
 
        isect_pix = np.where( identity > 0 )      # pixel coordinates of intersected pixels 
        isect_identity = identity[identity > 0]   # identity of intersected pixels 
        isect_pos = posi[identity > 0].copy()
        isect_pos[:,3] = 1.   ## replace identity with 1. so can apply transforms 

        isect_itheta = itheta_(isect_identity)
        isect_iphi   = iphi_(isect_identity)
        isect_index = index_(isect_identity)

        num_isect_pix = len(isect_identity)
        isect_fraction = float(num_isect_pix)/float(num_pix) 
        log.info("num_pix        : %d " % num_pix )
        log.info("num_isect_pix  : %d " % num_isect_pix )
        log.info("isect_fraction : %7.3f " % isect_fraction )

        self.posi = posi
        self.identity = identity

        # transform identity info for pixel intersects 
        self.isect_identity = isect_identity
        self.isect_pos = isect_pos  
        self.isect_iphi = isect_iphi
        self.isect_itheta = isect_itheta
        self.isect_index = isect_index

    def check_intersects(self):
        """
        #. hit counts onto each unique piece of intersected geometry 
        #. transform global positions into frame of the intersected geometry 
        """
        t = self
        t.u_isect_idx, t.u_isect_idx_count = np.unique(t.isect_index, return_counts=True) 

        for idx,count in zip(t.u_isect_idx, t.u_isect_idx_count):
            pos = t.isect_pos[t.isect_index == idx]  # global frame positions of intersects onto  
            assert len(pos) == count 
            lpos = np.dot( pos, t.it[idx-1] )       # apply inverse transform to global intersect positions  
            lpo = lpos[:,:3] 
            lra = np.sqrt(np.sum(lpo*lpo, axis=1))  # radius assuming intersects on a sphere  
            log.info("idx:%6d count:%6d lra.min:%10.4f lra.max:%10.4f " % (idx,count,lra.min(),lra.max())) 
        pass
         

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO) 
    np.set_printoptions(suppress=True, linewidth=200)
    np.set_printoptions(formatter={'int':hex}) 
    np.set_printoptions(formatter={'int':None}) 

    t = OSensorLibGeoTest()
    t.check_intersects()

