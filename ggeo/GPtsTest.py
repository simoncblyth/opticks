#!/bin/bash -l 

import logging, numpy as np

log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry
from opticks.ana.key import keydir

np.set_printoptions(linewidth=200, edgeitems=20)


strut370_midx_ = dict(new=96, old=91)



class GPts(object):
    """

    +---------+---------+----------+------------+
    |  lvIdx  |  ndIdx  |  csgIdx  |   ptIdx    |
    +---------+---------+----------+------------+

    * lvIdx and csgIdx always the same : they correspond to the midx 

    * ptIdx is 0-based pt index which is local to each ridx 

    * ndIdx is original 300k tree node index, instancing 
      means many gaps 

    * additional GPts.txt contains the boundary spec 

    ::

        In [3]: g[9].ipt                                                                                                                                                                                          
        Out[3]: 
        array([[  7,  10,   7,   0],
               [  6,  11,   6,   1],
               [  5,  12,   5,   2],
               [  4,  13,   4,   3],
               [  5,  14,   5,   4],
               [  4,  15,   4,   5],


    """
    def __init__(self, base, ridx):
        fold = Fold.Load(base, str(ridx))
        
        plc = fold.plcBuffer
        ipt = fold.iptBuffer

        assert len(plc) == len(ipt)

        self.base = base
        self.ridx = ridx
        self.fold = fold
        self.plc = plc
        self.ipt = ipt

    def __repr__(self):
        return "ridx %d plc %d ipt %d " % (self.ridx, len(self.plc), len(self.ipt) ) 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(linewidth=200, edgeitems=20)  

    kd = keydir(os.environ["OPTICKS_KEY"])
    arg = os.environ.get("OPTICKS_GEOCACHE_HOOKUP_ARG", None)

    cfdir = os.path.join(kd, "CSG_GGeo/CSGFoundry")
    cf = CSGFoundry(cfdir) if os.path.isdir(cfdir) else None


    gpdir = os.path.join(kd, "DebugGParts")
    gpts_dir = os.path.join(kd,"GPts")

    ridxs = list(map(int, sorted(os.listdir(os.path.expandvars(gpts_dir)))))

    log.info("kd    : %s " % kd ) 
    log.info("cfdir : %s " % cfdir) 
    log.info("gpdir : %s " % gpdir) 
    log.info("ridxs : %s " % str(ridxs) ) 
    log.info("g[0] ...")


    g = {} 
    for ridx in ridxs:
        g[ridx] = GPts(gpts_dir, ridx) 
    pass

    #expr = "g[0].plc.reshape(-1,16)"    
    #print(expr)
    #print(eval(expr))


    for ridx in ridxs:
        assert np.all( g[ridx].ipt[:,0] == g[ridx].ipt[:,2] )    
        ptidx_expect = np.arange(len(g[ridx].ipt)) 
        assert np.all( g[ridx].ipt[:,3] == ptidx_expect ) 
    pass


    #g[0].plc[np.where( g[0].ipt[:,0] == 96 )[0]]   

    # GPts placement transforms of the ridx solid selected by midx (aka lvIdx)
    ridx_midx_plc_ = lambda ridx,midx:g[ridx].plc[np.where( g[ridx].ipt[:,0] == midx)[0]]


    # need to use the midx from ipt to select the plc

    # BUT: the midx will change by geometry 


    
    strut370_midx = strut370_midx_[arg]
    print("arg:%s strut370_midx:%d " % (arg, strut370_midx))
   
    expr = "ridx_midx_plc_(0,%d).reshape(-1,16)" % strut370_midx
    print(expr)
    print(eval(expr))








