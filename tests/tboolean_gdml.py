#!/usr/bin/env python
"""
tboolean_gdml.py
=================

Usage::

   tboolean-

   tboolean-gdml-             
   # see logging from this script, parsing GDML and converting subtree into NCSG

   tboolean-gdml              
   # subtree visualization of polygonization and raytrace 

   tboolean-gdml-scan         
   # NCSGScanTest load single solid and SDF scan a line segment thru geometry

   tboolean-gdml-deserialize  
   # NCSGDeserializeTest all solids within subtree, eg 5 solids for /dd/Geometry/PMT/lvPmtHemi0x

   tboolean-gdml-ip
   # jump into ipython running this script to provide GDML tree ready for querying 

"""

import numpy as np
import os, logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.gdml import GDML

from opticks.dev.csg.csg import CSG  

if __name__ == '__main__':

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))


    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 
    gdml = GDML.parse(gdmlpath)
    tree = Tree(gdml.world, postype="position")

    subtree = tree.subtree(gsel, maxdepth=gmaxdepth, maxnode=gmaxnode, idx=gidx)


    im = dict(poly="IM", resolution="50")

    csgnodes = []
    for i, node in enumerate(subtree): 

        solid = node.lv.solid

        csgnode = solid.as_ncsg()

        if i > 0: # skip first node transform which is placement of targetNode within its parent 
            csgnode.transform = node.pv.transform
        pass 
        csgnode.meta.update(im)
        csgnode.boundary = args.testobject
        csgnodes.append(csgnode)
    pass


    container = CSG("box")
    container.boundary = args.container
    container.meta.update(poly="IM", resolution="40")
    container.meta.update(container="1", containerscale="4") 
    # container="1" meta data signals NCSG deserialization 
    # to adjust box size and position to contain contents 

    objs = []
    objs.append(container)
    objs.extend(csgnodes)

    for obj in objs:
        obj.dump()
    pass

    CSG.Serialize(objs, args.csgpath, outmeta=True )  


