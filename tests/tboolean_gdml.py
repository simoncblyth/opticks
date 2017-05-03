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
from opticks.ana.pmt.polyconfig import PolyConfig

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
    tree = Tree(gdml.world)

    subtree = tree.subtree(gsel, maxdepth=gmaxdepth, maxnode=gmaxnode, idx=gidx)

    log.info(" subtree %s nodes " % len(subtree) ) 

    cns = []
     
    for i, node in enumerate(subtree): 

        solid = node.lv.solid

        if i % 100 == 0:log.info("[%2d] converting solid %r " % (i,solid.name))

        polyconfig = PolyConfig(node.lv.shortname)

        cn = solid.as_ncsg()

        has_name = cn.name is not None and len(cn.name) > 0
        assert has_name, "\n"+str(solid)

        if i > 0: # skip first node transform which is placement of targetNode within its parent 
            cn.transform = node.pv.transform
        pass 
        cn.meta.update(polyconfig.meta )
        cn.meta.update(node.meta)

        cn.boundary = args.testobject
        cns.append(cn)
    pass


    container = CSG("box")
    container.boundary = args.container
    container.meta.update(PolyConfig("CONTAINER").meta)

    objs = []
    objs.append(container)
    objs.extend(cns)

    #for obj in objs: obj.dump()

    CSG.Serialize(objs, args.csgpath, outmeta=True )  

"""


* TODO: split the translation interface sensitivity from rotation sensitivity

::

   tboolean-gdml- --gsel 0 --gmaxdepth 2    # world is too big for easy navigation
   tboolean-gdml- --gsel 1 --gmaxdepth 1


"""
