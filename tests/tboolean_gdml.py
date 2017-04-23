#!/usr/bin/env python
"""
tboolean_gdml.py
=================

Usage::

   tboolean-
   tboolean-gdml-   # see the output of this script doing the conversion into NCSG
   tboolean-gdml    # visualize resulting polygonization and raytrace


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
    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 

    gdml = GDML.parse(gdmlpath)

    tree = Tree(gdml.world, postype="position")
    lvn = "/dd/Geometry/PMT/lvPmtHemi0x"

    allv = tree.findlv(lvn)     
    assert len(allv) == 672
    first = allv[0]

    progeny = first.rprogeny()
    assert len(progeny) == 5 

    container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="20")

    objs = []
    objs.append(container)

    for node in progeny: 
        csgnode = node.lv.solid.as_ncsg()
        csgnode.transform = node.pv.transform

        csgnode.dump()

        im = dict(poly="IM", resolution="50")
        csgnode.meta.update(im)
        csgnode.boundary = args.testobject

        objs.append(csgnode)
    pass

    CSG.Serialize(objs, args.csgpath, outmeta=True )  


