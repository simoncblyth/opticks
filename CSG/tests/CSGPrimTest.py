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
        pl.add_mesh(bx, style='wireframe', color=COLORS[i%len(COLORS)], line_width=5)
    pass


def make_boxplot(prims, crosshairs=0):
    pl = pv.Plotter(window_size=SIZE*2 )

    boxplot(pl, prims) 
   
    if crosshairs > 0.:
        CrossHairs.draw(pl, 6000)
    pass 

    pl.show_grid()
    return pl



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg = os.environ.get("OPTICKS_GEOCACHE_HOOKUP_ARG", None)
    kd = keydir(os.environ["OPTICKS_KEY"])
    cfdir = os.path.join(kd, "CSG_GGeo/CSGFoundry")

    log.info("arg   : %s " % arg ) 
    log.info("kd    : %s " % kd ) 
    log.info("cfdir : %s " % cfdir) 

    cf = CSGFoundry(cfdir) if os.path.isdir(cfdir) else None
    if not cf is None:
        print(cf)
    pass 

    solid_numPrim = cf.solid[:,1,0] 
    solid_primOffset = cf.solid[:,1,1]
    solid_type = cf.solid[:,1,2] ; assert np.all(solid_type == 0 ) 



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


    # prims that form a solid  (aka layers) picked by the ridx of the compound "solid" 
    ridx_prims_ = lambda ridx:cf.prim[solid_primOffset[ridx]:solid_primOffset[ridx]+solid_numPrim[ridx]]     

    # prims picked by meshindex (aka lvIdx)
    midx_prims_ = lambda midx:cf.prim[prim_meshIdx_(cf.prim) == midx]

    midx_prims_bbox_ = lambda midx:midx_prims_(midx)[:,2:].reshape(-1,8) 


    # prims picked by tranOffset 
    toff_prims_ = lambda toff:cf.prim[prim_tranOffset_(cf.prim) == toff]

    
    prim_numNode         = prim_numNode_(cf.prim)
    prim_nodeOffset      = prim_nodeOffset_(cf.prim)
    prim_tranOffset      = prim_tranOffset_(cf.prim)
    prim_planOffset      = prim_planOffset_(cf.prim) ; assert np.all( prim_planOffset == 0 ), "no planOffset"   

    prim_sbtIndexOffset =  prim_sbtIndexOffset_(cf.prim)
    prim_meshIdx        =  prim_meshIdx_(cf.prim)
    prim_repeatIdx      =  prim_repeatIdx_(cf.prim)
    prim_primIdx        =  prim_primIdx_(cf.prim)
 
    prim_meshIdx_to_numNode = dict(zip(prim_meshIdx,prim_numNode))  


    mname_skip = "Flange"
    all_ridxs = list(range(0,len(cf.solid)))
    ridxs = all_ridxs if not "RIDX" in os.environ else list(map(int, os.environ["RIDX"].split(",")))

    print("\n\n all_ridxs: %s   ridxs:%s   nmame_skip:%s   geocache_hookup_arg:%s " % (str(all_ridxs), str(ridxs), mname_skip, arg ))
    skip = 0 
    for ridx in ridxs:
        ridx_prims = ridx_prims_(ridx)

        print("\n ridx : %2d   ridx_prims.shape %s " % (ridx, str(ridx_prims.shape))) 

        midx = prim_meshIdx_(ridx_prims)   
        numn = prim_numNode_(ridx_prims) 

        u_mx, c_mx = np.unique(midx, return_counts=True)  
        print(" %4s %4s %4s : %60s : %s " % ("u_mx","c_mx", "nnd", " unique midx prim counts and meshname ", "prs.shape"  ))
        for i in range(len(u_mx)):
            sel = prim_meshIdx_(cf.prim) == u_mx[i]    
            prs = cf.prim[sel] 
            mname = cf.meshname[u_mx[i]]
            mx = u_mx[i]   # mesh index
            cn = c_mx[i]   # number of prims with the mesh index
            nnd = prim_meshIdx_to_numNode[mx]   ## number of nodes of the prim 

            bb = midx_prims_bbox_(mx)  

            if mname.find(mname_skip) > -1: 
                skip += 1 
                continue  
            pass
            print(" %4d %4d %4d : %60s : %20s : %s  "  % (mx, cn, nnd, mname, str(prs.shape), str(bb[0]) ))
        pass
    pass
    print(" skip:%d  nmame_skip:%s " % (skip, mname_skip))


    #pl = make_boxplot( ridx_prims_(9) )
    #pl.show()    

    # meshnames of prim with tranOffset zero  
    #cf.meshname[prim_meshIdx_(toff_prims_(0))]   



