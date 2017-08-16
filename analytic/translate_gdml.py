#!/usr/bin/env python
"""
translate_gdml.py
=================

Usage::

   tboolean-

   tboolean-0-             
   tboolean-gds-             
   tboolean-oav-             
   tboolean-iav-             
   tboolean-pmt-             
       see logging from this script, parsing GDML and translating subtree into NCSG


   tboolean-gdml- --gsel 0 --gmaxdepth 2    # world is too big for easy navigation
   tboolean-gdml- --gsel 1 --gmaxdepth 1
   
   tboolean-0
   tboolean-gds             
   tboolean-oav             
   tboolean-iav             
   tboolean-pmt             
       subtree visualization of polygonization and raytrace 

   tboolean-gdml-scan         
   # NCSGScanTest load single solid and SDF scan a line segment thru geometry

   tboolean-0-deserialize  
   # NCSGDeserializeTest all solids within subtree, eg 5 solids for /dd/Geometry/PMT/lvPmtHemi0x

   tboolean-gdml-ip
   # jump into ipython running this script to provide GDML tree ready for querying 



* TODO: split the translation interface sensitivity from rotation sensitivity


"""

import numpy as np
import os, logging, sys
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.analytic.treebase import Tree
from opticks.analytic.gdml import GDML
from opticks.analytic.polyconfig import PolyConfig
from opticks.analytic.csg import CSG  
from opticks.analytic.sc import Sc


if __name__ == '__main__':

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))


    gdml = GDML.parse()
    tree = Tree(gdml.world)

    subtree = tree.subtree(gsel, maxdepth=gmaxdepth, maxnode=gmaxnode, idx=gidx)
    # just a flat list of nodes

    log.info(" subtree %s nodes " % len(subtree) ) 

    cns = []
     
    for i, node in enumerate(subtree): 

        solid = node.lv.solid
        if i % 100 == 0:log.info("[%2d] converting solid %r " % (i,solid.name))

        cn = Sc.translate_lv(node.lv, maxcsgheight=4)

        skip_transform = i == 0  # skip first node transform which is placement of targetNode within its parent 
        if skip_transform:
            log.warning("skipping transform")
        else: 
            cn.transform = node.pv.transform
        pass 

        cn.meta.update(node.meta)

        log.info("cn.meta %r " % cn.meta )

        sys.stderr.write("\n".join(["","",solid.name,repr(cn),str(cn.txt)])) 

        cn.boundary = args.testobject
        cn.node = node

        cns.append(cn)
    pass


    container = None
    if args.disco:
        log.info("--disco option skipping container")
    else:
        log.info("adding container")
        container = CSG("box", name="container")
        container.boundary = args.container
        container.meta.update(PolyConfig("CONTAINER").meta)
    pass

    objs = []
    if not container is None:
        objs.append(container)
    pass
    objs.extend(cns)

    CSG.Serialize(objs, args.csgpath, outmeta=True )  


