#!/usr/bin/env python
"""
tboolean_pmt.py
=================

Usage::

   tboolean-
   tboolean-pmt-   # see the output of this script doing the conversion into NCSG
   tboolean-pmt    # visualize resulting polygonization and raytrace


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

from opticks.dev.csg.csg import CSG  





DEFAULT = "DEFAULT"
PYREX = "lvPmtHemi"    
VACUUM = "lvPmtHemiVacuum"   
CATHODE = "lvPmtHemiCathode"
BOTTOM = "lvPmtHemiBottom"
DYNODE = "lvPmtHemiDynode"

class ConfigPoly(object):
    """
    Common location for volume specific polygonization settings
    to avoid duplication between the GDML and detdesc branches.
    """

    _seeds = {
        CATHODE: "0,0,127.9,0,0,1",
         BOTTOM:"0,0,0,0,0,-1", 
     }

    _resolution = {
          BOTTOM:"150",
         DEFAULT:"40",
    }

    _verbosity = {
          DYNODE:"3",
         DEFAULT:"0"
    }

    def __init__(self, lvn):
        self.lvn = lvn 
    
    resolution = property(lambda self:self._resolution.get(self.lvn, self._resolution[DEFAULT]))
    verbosity = property(lambda self:self._verbosity.get(self.lvn, self._verbosity[DEFAULT]))
    seeds = property(lambda self:self._seeds.get(self.lvn, self._seeds[DEFAULT]))




if __name__ == '__main__':

    args = opticks_main()

    g = Dddb.parse(args.apmtddpath)

    #lvn = "lvPmtHemiwPmtHolder"
    lvn = "lvPmtHemi" 

    lv = g.logvol_(lvn)
    tr = Tree(lv)
    nn = tr.num_nodes()

    if lvn == "lvPmtHemi":
        assert nn == 5
    pass


    container = CSG("box", param=[0,0,0,1000], boundary=args.container, poly="IM", resolution="20")

    objs = []
    objs.append(container)


    im = dict(poly="IM")
    mc = dict(poly="MC", nx="30")
    dcs = dict(poly="DCS", nominal="7", coarse="7", threshold="1", verbosity="0")

    #ii = [DYNODE]
    ii = range(nn)


    for i in ii:

        node = tr.get(i)
        lv = node.lv 
        lvn = lv.name

        cfg = ConfigPoly(lv.shortname)


        log.info("\ntranslating .............lvn %s ....node  %r " % (lvn, node) )


        seeds = vseeds.get(lvn, None)
        if seeds is not None:  im.update(seeds=seeds)

        resolution = vresolution.get(lvn, vresolution[DEFAULT])
        verbosity = vverbosity.get(lvn, vverbosity[DEFAULT])
        im.update(resolution=resolution, verbosity=verbosity)

        if lvn in [CATHODE, BOTTOM]:
            #poly = dcs
            poly = im
        else:
            poly = im
        pass


        obj = NCSGConverter.ConvertLV( lv )
        obj.boundary = args.testobject
        obj.meta.update(poly) 
        obj.meta.update(gpuoffset="-200,0,0")  # shift raytrace only up in x
        
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



