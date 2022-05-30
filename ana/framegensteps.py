#!/usr/bin/env python
"""
framegensteps.py
=================

Uses per-genstep transforms to give world_frame_centers and 
then uses the frame.w2m to get the grid centers in the target frame.  


Related:

* sysrap/sframe.h 
* sysrap/SFrameGenstep.cc:SFrameGenstep::MakeCenterExtentGensteps


"""
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
    def __init__(self, genstep, frame, local=True):
        """
        :param genstep: (num_gs,6,4) array with grid transforms in 2: and position in 1 
        :param frame: sframe instance (replacing former metatran array of 3 transforms and grid GridSpec instance)
        :param local: bool
        :param local_extent_scale: SUSPECT THIS SHOULD NO LONGER EVER BE TRUE

        The way *local* is used implies that the simtrace genstep in q1 contains global centers, 
        that are transformed here by frame.w2m to give the genstep centers *ugsc* in the local frame  

        BUT t.genstep[:,1] they are all origins [0., 0., 0., 1.]

        YES, but the genstep transfrom is applied to that origin to give the world_frame_centers 
        hence need the frame.w2m transform to give the local frame grid centers. 

        """
        gs = genstep
        local_extent_scale = frame.coords == "RTP"  ## KINDA KLUDGE DUE TO EXTENT HANDLING BEING DONE BY THE RTP TRANSFORM

        numpho = gs.view(np.int32)[:,0,3]        # q0.i.w  top right values from all gensteps
        gsid = gs.view(np.int32)[:,0,2].copy()   # q0.i.z  SEvent::ConfigureGenstep
        all_one = np.all( gs[:,1,3] == 1. )      # q1.f.w 
        assert all_one   # from SEvent::MakeCenterExtentGensteps that q1.f.w should always to 1.f


        ## apply the 4x4 transform in rows 2: to the position in row 1 
        world_frame_centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
        for igs in range(len(gs)): 
            gs_pos = gs[igs,1]          ## normally origin (0,0,0,1)
            gs_tran = gs[igs,2:]        ## m2w with grid translation 
            world_frame_centers[igs] = np.dot( gs_pos, gs_tran )    
            #   world_frame_centers = m2w * grid_translation * model_frame_positon
        pass

        centers_local = np.dot( world_frame_centers, frame.w2m )  
        # use w2m to transform global frame centers back to local frame

        if local and local_extent_scale:
            extent = frame.ce[3]
            centers_local[:,:3] *= extent    
            assert 0 
            ## HUH:confusing, surely the horse has left the stable ?
            ## what was done on device is what matters, so why scale here ?
            ## vague recollection this is due to some issue with RTP tangential transforms
            ## TODO: ELIMINATE once get RTP tangential operational again
        pass


        ugsc = centers_local if local else world_frame_centers  

        lim = {}
        lim[X] = np.array([ugsc[:,X].min(), ugsc[:,X].max()])
        lim[Y] = np.array([ugsc[:,Y].min(), ugsc[:,Y].max()])  
        lim[Z] = np.array([ugsc[:,Z].min(), ugsc[:,Z].max()])  

        self.gs = gs
        self.gsid = gsid
        self.numpho = numpho
        self.totpho = np.sum(numpho)
        self.centers = world_frame_centers 
        self.centers_local = centers_local
        self.ugsc = ugsc
        self.lim = lim 
        log.info(repr(self))

    def __repr__(self):
        return "\n".join([
                   "FrameGensteps gs.gs %s " % str(self.gs.shape),
                   "gs.numpho[0] %d " % self.numpho[0] ,
                   "gs.totpho    %d " % self.totpho ,
                   "gs.lim[X] %s " % str(self.lim[X]),
                   "gs.lim[Y] %s " % str(self.lim[Y]),
                   "gs.lim[Z] %s " % str(self.lim[Z]),
              ])



