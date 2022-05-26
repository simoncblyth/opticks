#!/usr/bin/env python

import os, numpy as np
import logging 
from opticks.ana.axes import X, Y, Z

log = logging.getLogger(__name__)


class Positions(object):
    """
    Transforms global intersect positions into local frame 

    HMM: the local frame positions are model frame in extent units  
    when using tangential ... not so convenient would be better 
    with real mm dimensions in local : kludge this with local_extent_scale=True

    The isect, gpos used here come from qevent::add_simtrace
    """

    @classmethod
    def Check(cls, p):
        num_photon = len(p)
        distance = p[:,0,3]
        landing_count = np.count_nonzero( distance )
        landing_msg = "ERROR NO PHOTON LANDED" if landing_count == 0 else ""
        print(" num_photon: %d : landing_count : %d   %s " % (num_photon, landing_count, landing_msg) )


    def __init__(self, p, gs, frame, local=True, mask="pos", local_extent_scale=False ):
        """
        :param p: photons array  (should be called "simtrace" really)
        :param gs: Gensteps instance
        :param frame: formerly GridSpec instance 
        """
        isect = p[:,0]

        gpos = p[:,1].copy()            # global frame intersect positions
        gpos[:,3] = 1  

        lpos = np.dot( gpos, frame.w2m )   # local frame intersect positions

        if local and local_extent_scale:
            extent = frame.ce[3]
            lpos[:,:3] *= extent 
        pass

        upos = lpos if local else gpos

        poslim = {}
        poslim[X] = np.array([upos[:,X].min(), upos[:,X].max()])
        poslim[Y] = np.array([upos[:,Y].min(), upos[:,Y].max()])  
        poslim[Z] = np.array([upos[:,Z].min(), upos[:,Z].max()])  

        self.poslim = poslim 
        self.gs = gs
        self.frame = frame 

        self.isect = isect
        self.gpos = gpos
        self.lpos = lpos
        self.local = local

        ## NB when applying the mask the below are changed  
        self.p = p 
        self.upos = upos

        #self.make_histogram()

        if mask == "pos":
            self.apply_pos_mask()
        elif mask == "t":
            self.apply_t_mask()
        else: 
            pass
        pass

    def apply_mask(self, mask):
        """
        applying the mask changes makes a selection on the self.p and self.upos arrays
        which will typically decrease their sizes
        """
        self.mask = mask
        self.p = self.p[mask]
        self.upos = self.upos[mask]


    def apply_pos_mask(self):
        """
        pos_mask restricts upos positions to be within the limits defined by 
        the source genstep positions
        """
        lim = self.gs.lim  

        xmin, xmax = lim[0] 
        ymin, ymax = lim[1] 
        zmin, zmax = lim[2] 

        upos = self.upos

        xmask = np.logical_and( upos[:,0] >= xmin, upos[:,0] <= xmax )
        ymask = np.logical_and( upos[:,1] >= ymin, upos[:,1] <= ymax )
        zmask = np.logical_and( upos[:,2] >= zmin, upos[:,2] <= zmax )

        xy_mask = np.logical_and( xmask, ymask )
        xyz_mask = np.logical_and( xy_mask, zmask )
        mask = xyz_mask 

        log.info("apply_pos_mask")
        self.apply_mask(mask)

    def apply_t_mask(self):
        """
        t_mask restricts the intersect distance t to be greater than zero
        this excludes misses 
        """
        log.info("apply_t_mask")
        t = self.p[:,2,2]
        mask = t > 0. 
        self.apply_mask( mask) 

    def make_histogram(self):
        """
        TODO: use  3d histo like this to sparse-ify gensteps positions, 
        to avoiding shooting rays from big voids 
        """
        lim = self.gs.lim  
        nx = self.frame.nx
        ny = self.frame.ny
        nz = self.frame.nz
        upos = self.upos

        # bizarrely some python3 versions think the below are SyntaxError without the num= 
        #      SyntaxError: only named arguments may follow *expression
        
        binx = np.linspace(*lim[X], num=2*nx+1)
        biny = np.linspace(*lim[Y], num=max(2*ny+1,2) )
        binz = np.linspace(*lim[Z], num=2*nz+2)

        bins = ( binx, biny, binz )

        h3d, bins2 = np.histogramdd(upos[:,:3], bins=bins )   

        self.h3d = h3d
        self.bins = bins 
        self.bins2 = bins2 



