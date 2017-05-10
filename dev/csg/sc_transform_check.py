#!/usr/bin/env python
"""

In [4]: txf[:,3,2]
Out[4]: 
array([-1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750., -1750.,
       -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250., -1250.,
        -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,  -750.,
        -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,  -250.,
         250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,   250.,
         750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,   750.,
        1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,  1250.,
        1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.,  1750.], dtype=float32)


"""
import numpy as np
import os, logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.gdml import GDML
from opticks.ana.pmt.treebase import Tree
from opticks.dev.csg.sc import Sc


if __name__ == '__main__':

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    gsel = "/dd/Geometry/AD/lvSST0x" 
    gmaxdepth = 3

    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))

    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 
    gdml = GDML.parse(gdmlpath)


    tree = Tree(gdml.world)
    target = tree.findnode(gsel, gidx)

    log.info(" target node %s " % target )   


    sc = Sc()
    tg = sc.add_tree_gdml( target, maxdepth=3 )
    gltf = sc.save("$TMP/nd/scene.gltf")



    lvIdx = sc.find_meshes_so("pmt-hemi")[0].lvIdx    
    pmts = tg.find_nodes(lvIdx)
    assert len(pmts) == 192 == 24*8

    txf = np.vstack([pmt.transform for pmt in pmts]).reshape(8,24,4,4)




