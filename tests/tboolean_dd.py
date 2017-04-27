#!/usr/bin/env python
"""
tboolean_dd.py
=================

Usage::

   tboolean-
   tboolean-dd-   # see the output of this script doing the conversion into NCSG
   tboolean-dd    # visualize resulting polygonization and raytrace


#. FIXED: solid-0:PYREX poly-cylinder not matching raytrace, NCylinder SDF was ignoring center.z and bbox was wrong 

#. FIXED: solid-2:CATHODE restricting to just the inner observe the correct union of zsphere shape : but it disappears from the other side ?  
          this wierdness was due to using CSG sub-objects that are not solids : ie closed geometry, 
          endcaps to close sub-objects are required when using CSG 

#. FIXED: doing all together, are missing a translate transform for BOTTOM, the tranOffset needed passing in to csg_intersect_part


#. ISSUE: CATHODE is 0.05 mm thick diff of zspheres, this causes difficulties

   * finding the surface, needs manual seeding and extreme resolution (4000)
   * extreme resolution 4000, did not run into memory problems only due to continuation failure
   * without continuation failure even resolution ~200 runs into memory issues,
     and in any case that kinda resolution produces far too many tris
   * IM continuation fails to stay with the surface, only producing a very small mesh patch
  
#. ISSUE: BOTTOM is 1 mm thick diff of zspheres, similar issue to CATHODE but x20 less bad

   * IM succeeds with resolution 150, but too many tris to be of much use  
   * DCS nominal 7, coarse 7/6 produces a whacky partial mesh with "Moire lace pattern"  
   * DCS nominal 8, coarse 7 : slow and produces too many tris 670198, they are disqualified for being beyond bbox



"""

import numpy as np
import logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.ddbase import Dddb
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.ncsgconverter import NCSGConverter
from opticks.ana.pmt.polyconfig import PolyConfig

from opticks.dev.csg.csg import CSG  



if __name__ == '__main__':

    args = opticks_main()

    g = Dddb.parse(args.apmtddpath)

    lvn = "lvPmtHemi" 

    lv = g.logvol_(lvn)
    tr = Tree(lv)
    nn = tr.num_nodes()


    container = CSG("box")
    container.boundary = args.container
    container.meta.update(PolyConfig("CONTAINER").meta)

    objs = []
    objs.append(container)

    ii = range(nn)

    for i in ii:

        node = tr.get(i)
        lv = node.lv 
        lvn = lv.name

        pc = PolyConfig(lv.shortname)

        log.info("\ntranslating .............lvn %s ....node  %r " % (lvn, node) )

        obj = NCSGConverter.ConvertLV( lv )
        obj.boundary = args.testobject
        obj.meta.update(pc.meta)
        obj.meta.update(gpuoffset="-200,0,0")  # shift raytrace only 
        
        obj._translate += np.array([200, (i-2)*200, 0], dtype=np.float32 ) 

        # solid parade in Y (left-right) and shift everything down in Z  
        # translate applied to root nodes, gets propagated 
        # down to gtransforms on the primitives at serialization 

        log.info("obj.translate: %s " % (obj.translate) )
        log.info("obj.transform: \n%s " % (obj.transform) )

        objs.append(obj)

        CSG.Dump(obj)
    pass

    CSG.Serialize(objs, args.csgpath, outmeta=True )  
    # outmeta: stdout communicates to tboolean-- THAT MEANS NO OTHER "print" use logging instead



