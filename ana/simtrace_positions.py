#!/usr/bin/env python

import os, numpy as np
import logging 
from opticks.ana.axes import X, Y, Z

log = logging.getLogger(__name__)


class SimtracePositions(object):
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


    def __init__(self, simtrace, gs, frame, local=True, mask="pos", symbol="t_pos" ):
        """
        :param simtrace: "photons" array  
        :param gs: FrameGensteps instance, used for gs.lim position masking 
        :param frame: sframe instance
        :param local:
        :param mask:

       
        The simtrace array is populated by:

        1. cx/CSGOptiX7.cu:simtrace 
        2. CPU version of this ?

        271 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
        272 {
        ...
        274     sevent* evt  = params.evt ;
        280     const quad6& gs     = evt->genstep[genstep_id] ;
        281 
        282     qsim* sim = params.sim ;
        283     curandState rng = sim->rngstate[idx] ;
        284 
        285     quad4 p ;
        286     sim->generate_photon_simtrace(p, rng, gs, idx, genstep_id );
        287 
        288     const float3& pos = (const float3&)p.q0.f  ;
        289     const float3& mom = (const float3&)p.q1.f ;
        290 
        291     trace(
        292         params.handle,
        293         pos,
        294         mom,
        295         params.tmin,
        296         params.tmax,
        297         prd
        298     );
        299 
        300     evt->add_simtrace( idx, p, prd, params.tmin );
        301 
        302 }

        410 SEVENT_METHOD void sevent::add_simtrace( unsigned idx, const quad4& p, const quad2* prd, float tmin )
        411 {
        412     float t = prd->distance() ;  // q0.f.w 
        413     quad4 a ;  
        414     
        415     a.q0.f  = prd->q0.f ;
        416     
        417     a.q1.f.x = p.q0.f.x + t*p.q1.f.x ;
        418     a.q1.f.y = p.q0.f.y + t*p.q1.f.y ;
        419     a.q1.f.z = p.q0.f.z + t*p.q1.f.z ;
        420     a.q1.i.w = 0.f ;  
        421     
        422     a.q2.f.x = p.q0.f.x ;
        423     a.q2.f.y = p.q0.f.y ;
        424     a.q2.f.z = p.q0.f.z ;
        425     a.q2.u.w = prd->boundary() ; // was tmin, but expecting bnd from CSGOptiXSimtraceTest.py:Photons
        426     
        427     a.q3.f.x = p.q1.f.x ;
        428     a.q3.f.y = p.q1.f.y ;
        429     a.q3.f.z = p.q1.f.z ;
        430     a.q3.u.w = prd->identity() ;  // identity from __closesthit__ch (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) 
        431     
        432     simtrace[idx] = a ;
        433 }
        """
        local_extent_scale = frame.coords == "RTP"  ## KINDA KLUDGE DUE TO EXTENT HANDLING BEING DONE BY THE RTP TRANSFORM
        isect = simtrace[:,0]

        gpos = simtrace[:,1].copy()              # global frame intersect positions
        gpos[:,3] = 1.  

        lpos = np.dot( gpos, frame.w2m )   # local frame intersect positions

        if local and local_extent_scale:
            extent = frame.ce[3]
            lpos[:,:3] *= extent 
        pass

        upos = lpos if local else gpos   # local usually True 

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
        ## note that need to exclude the nan for comparisons to work howver
        ##      np.all( t_pos.simtrace[:,:3] == t.simtrace[:,:3] ) 
        self.simtrace = simtrace 
        self.upos = upos

        #self.make_histogram()

        if mask == "pos":
            self.apply_pos_mask()
        elif mask == "t":
            self.apply_t_mask()
        else: 
            pass
        pass
        self.symbol = symbol


    def __repr__(self):
        symbol = self.symbol
        return "\n".join([
                   "SimtracePositions",
                   "%s.simtrace %s " % (symbol, str(self.simtrace.shape)),
                   "%s.isect %s " % (symbol, str(self.isect.shape)),
                   "%s.gpos %s " % (symbol, str(self.gpos.shape)),
                   "%s.lpos %s " % (symbol, str(self.lpos.shape)),
                   ])
 
                   

    def apply_mask(self, mask):
        """
        applying the mask changes makes a selection on the self.p and self.upos arrays
        which will typically decrease their sizes
        """
        self.mask = mask
        self.simtrace = self.simtrace[mask]
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
        #t = self.simtrace[:,2,2]
        t = self.simtrace[:,0,3]
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


if __name__ == '__main__':
    pass

