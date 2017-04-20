#!/usr/bin/env python
"""
tboolean_pmt.py
=================

Usage::

   tboolean-
   tboolean-pmt-   # see the output of this script doing the conversion into NCSG
   tboolean-pmt    # visualize resulting polygonization and raytrace


#. FIXED: solid-0:PYREX poly-cylinder not matching raytrace, NCylinder SDF was ignoring center.z and bbox was wrong 

#. ISSUE: solid-2:CATHODE has two very close sphere shells, this is difficult for meshers to handle without very high resolution
          investigating in tboolean-difference-zsphere

#. FIXED: solid-2:CATHODE restricting to just the inner observe the correct union of zsphere shape : but it disappears from the other side ?  
          this wierdness was due to using CSG sub-objects that are not solids : ie closed geometry, 
          endcaps to close sub-objects are required when using CSG 

#. FIXED: doing all together, are missing a translate transform for BOTTOM, the tranOffset needed passing in to csg_intersect_part

"""

import logging
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.ddbase import Dddb
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.ncsgconverter import NCSGConverter

from opticks.dev.csg.csg import CSG  

if __name__ == '__main__':

    args = opticks_main()

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")
    tr = Tree(lv)
    nn = tr.num_nodes()
    assert nn == 5


    container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="20")

    objs = []
    objs.append(container)

    im = dict(poly="IM", resolution="500", verbosity="2")
    mc = dict(poly="MC", nx="30")
    dcs = dict(poly="DCS", nominal="7", coarse="6", threshold="1", verbosity="0")
    poly = im

    PYREX = 0    # raytrace + poly OK
    VACUUM = 1   # raytrace + poly OK
    CATHODE = 2  # raytrace OK when enable endcaps (othersize unions of zspheres wierdness), poly fails 
    BOTTOM = 3   # raytrace OK, poly fails (does not find inside d)  : tris outside box?
    DYNODE = 4   # looks OK

    ii = [CATHODE]
    #ii = range(nn)

    for i in ii:
        root = tr.get(i)
        log.info("\ntranslating ..........................  %r " % root )

        obj = NCSGConverter.ConvertLV( root.lv )
        obj.boundary = args.testobject
        obj.meta.update(poly)  
        objs.append(obj)

        CSG.Dump(obj)
    pass

    CSG.Serialize(objs, args.csgpath, outmeta=True )  
    # outmeta: stdout communicates to tboolean-- THAT MEANS NO OTHER "print" use logging instead



