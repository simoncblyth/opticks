#!/usr/bin/env python

import os, logging, numpy as np
from opticks.ana.axes import X,Y,Z

log = logging.getLogger(__name__)

class FrameGensteps(object):
    """
    Transform enabled gensteps:

    * gs[igs,0,3] photons to generate for genstep *igs* 
    * gs[igs,1] local frame center position
    * gs[igs,2:] 4x4 transform  

    From SEvent::ConfigureGenstep::

    * gsid was MOVED from (1,3) to (0,2) when changing genstep to carry transform

    Notice that every genstep has its own transform with slightly 
    different translations according to the different grid points 
    Conversely there is only one overall frame transform
    which corresponds to the targetted piece of geometry.
    """
    def __init__(self, genstep, frame, local=True, local_extent_scale=False ):
        """
        :param genstep: (num_gs,6,4) array with grid transforms in 2: and position in 1 
        :param frame: sframe instance replacing former metatran array of 3 transforms and grid GridSpec instance

        """
        gs = genstep

        numpho = gs.view(np.int32)[:,0,3]  # top right values from all gensteps
        gsid = gs.view(np.int32)[:,0,2].copy()  # SEvent::ConfigureGenstep
        all_one = np.all( gs[:,1,3] == 1. ) 
        assert all_one   # from SEvent::MakeCenterExtentGensteps that q1.f.w should always to 1.f

        ## apply the 4x4 transform in rows 2: to the position in row 1 
        centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
        for igs in range(len(gs)): 
            centers[igs] = np.dot( gs[igs,1], gs[igs,2:] )  
        pass

        tran = frame.w2m 
        centers_local = np.dot( centers, tran )  # use metatran.v to transform back to local frame

        if local and local_extent_scale:
            extent = frame.ce[3]
            centers_local[:,:3] *= extent 
        pass

        ugsc = centers_local if local else centers  

        lim = {}
        lim[X] = np.array([ugsc[:,X].min(), ugsc[:,X].max()])
        lim[Y] = np.array([ugsc[:,Y].min(), ugsc[:,Y].max()])  
        lim[Z] = np.array([ugsc[:,Z].min(), ugsc[:,Z].max()])  

        self.gs = gs
        self.gsid = gsid
        self.numpho = numpho
        self.centers = centers 
        self.centers_local = centers_local
        self.ugsc = ugsc
        self.lim = lim 

        log.info("Gensteps\n %s " % repr(self))


    def __repr__(self):
        return "\n".join([
                   "gs.gs %s " % str(self.gs.shape),
                   "gs.numpho %s " % self.numpho,
                   "gs.lim[X] %s " % str(self.lim[X]),
                   "gs.lim[Y] %s " % str(self.lim[Y]),
                   "gs.lim[Z] %s " % str(self.lim[Z]),
              ])



