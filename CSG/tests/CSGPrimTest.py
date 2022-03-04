#!/usr/bin/env python

import os, logging, numpy as np
np.set_printoptions(edgeitems=16)  

from opticks.ana.key import keydir
from opticks.ana.gridspec import CrossHairs
from opticks.CSG.CSGFoundry import CSGFoundry

log = logging.getLogger(__name__)

SIZE = np.array([1280, 720])

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

COLORS = "red green blue cyan magenta yellow pink violet".split()



def boxplot(pl, ridx_prims):
    """
    ::

          pr[2,0] pr[2,1] pr[2,2]         
          pr[2,3] pr[3,0] pr[3,1]

    """
    for i in range(len(ridx_prims)):
        pr = ridx_prims[i]
        bb = (pr[2,0], pr[2,3],  pr[2,1], pr[3,0], pr[2,2], pr[3,1] )
        bx = pv.Box(bounds=bb)
        pl.add_mesh(bx, style='wireframe', color=COLORS[i%len(colors)], line_width=5)
    pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    kd = keydir(os.environ["OPTICKS_KEY"])
    cfdir = os.path.join(kd, "CSG_GGeo/CSGFoundry")

    log.info("kd    : %s " % kd ) 
    log.info("cfdir : %s " % cfdir) 

    cf = CSGFoundry(cfdir) if os.path.isdir(cfdir) else None

    solid_numPrim = cf.solid[:,1,0] 
    solid_primOffset = cf.solid[:,1,1]
    solid_type = cf.solid[:,1,2] ; assert np.all(solid_type == 0 ) 

    # prims of solid picked by ridx 
    prims_ = lambda ridx:cf.prim[solid_primOffset[ridx]:solid_primOffset[ridx]+solid_numPrim[ridx]]     


    prim_bounds = np.zeros( (len(cf.prim), 6), dtype=np.float32 )
    prim_bounds[:,0] = cf.prim[:,2,0]
    prim_bounds[:,1] = cf.prim[:,2,3]
    prim_bounds[:,2] = cf.prim[:,2,1]
    prim_bounds[:,3] = cf.prim[:,3,0]
    prim_bounds[:,4] = cf.prim[:,2,2]
    prim_bounds[:,5] = cf.prim[:,3,1]

    node_tran = cf.node.view(np.int32)[:,3,3] & 0x7fffffff  
 
    prim_numNode_        = lambda a:a.view(np.int32)[:,0,0]    
    prim_nodeOffset_     = lambda a:a.view(np.int32)[:,0,1]    
    prim_tranOffset_     = lambda a:a.view(np.int32)[:,0,2]    
    prim_planOffset_     = lambda a:a.view(np.int32)[:,0,3]    

    prim_sbtIndexOffset_ = lambda a:a.view(np.int32)[:,1,0]    
    prim_meshIdx_        = lambda a:a.view(np.int32)[:,1,1]    
    prim_repeatIdx_      = lambda a:a.view(np.int32)[:,1,2]
    prim_primIdx_        = lambda a:a.view(np.int32)[:,1,3]

    
    prim_numNode         = prim_numNode_(cf.prim)
    prim_nodeOffset      = prim_nodeOffset_(cf.prim)
    prim_tranOffset      = prim_tranOffset_(cf.prim)
    prim_planOffset      = prim_planOffset_(cf.prim) ; assert np.all( prim_planOffset == 0 ), "no planOffset"   

    prim_sbtIndexOffset =  prim_sbtIndexOffset_(cf.prim)
    prim_meshIdx        =  prim_meshIdx_(cf.prim)
    prim_repeatIdx      =  prim_repeatIdx_(cf.prim)
    prim_primIdx        =  prim_primIdx_(cf.prim)
 


    for ridx in range(0,len(cf.solid)):
        ridx_prims = prims_(ridx)

        print("\n\n ridx : %2d   ridx_prims.shape %s " % (ridx, str(ridx_prims.shape))) 

        midx = prim_meshIdx_(ridx_prims)   

        u_mx, c_mx = np.unique(midx, return_counts=True)  
        print(" %4s %4s : %60s : %s " % ("u_mx","c_mx", " unique midx prim counts and meshname ", "prs.shape"  ))
        for i in range(len(u_mx)):
            sel = prim_meshIdx_(cf.prim) == u_mx[i]    
            prs = cf.prim[sel] 
            print(" %4d %4d : %60s : %s "  % (u_mx[i], c_mx[i], cf.meshname[u_mx[i]], str(prs.shape) ))
        pass
    pass


    # select prims with a meshindex (aka lvIdx)
    midx_prims_ = lambda midx:cf.prim[prim_meshIdx_(cf.prim) == midx]




    if 0:
        pl = pv.Plotter(window_size=SIZE*2 )

        boxplot(pl, ridx_prims) 
        #CrossHairs.draw(pl, 6000)

        pl.show_grid()
        pl.show()    
    pass




