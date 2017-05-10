#!/usr/bin/env python
"""

"""
import numpy as np
import os, logging

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.nbase import find_ranges
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.gdml import GDML
from opticks.ana.pmt.polyconfig import PolyConfig
from opticks.dev.csg.textgrid import TextGrid

from opticks.dev.csg.csg import CSG  


class SNode(object):
    """
    Solid node for establishing relationships between all solids within 
    a geometry, especially looking for top solids that are not
    contained within any other solid.
    """
    registry = {}

    @classmethod
    def create(cls, idx, ssidx=[]):
        sn = SNode(idx, ssidx)
        cls.registry[idx] = sn 
        log.debug("SNode.create %r " % (sn))
        for ssi in ssidx:
            if ssi == idx:continue
            child = cls.registry[ssi]
            assert child
            child.top = sn  
            # top for some SNode will change as more SNode are created, 
            # but dont care, are just looking for SNode without tops
        pass
        return sn

    @classmethod
    def tops(cls, ssmin=0):
        return filter(lambda sn:sn.top is None and sn.ssn >= ssmin, cls.registry.values())

    def __repr__(self):
        return "%3d; %s " % (self.idx, self.ssd)

    def __init__(self, idx, ssidx ):
        self.idx = idx
        self.ss = ssidx
        ssr = list(find_ranges(sorted(ssidx)))
        assert len(ssr) in (0,1)  
        if len(ssr) == 1:
            ssr = ssr[0]
            ssn = ssr[1] - ssr[0] + 1 
            ssd = "%2d:%3d-%3d" % (ssn, ssr[0], ssr[1])
        elif len(ssr) == 0:
            ssr = None
            ssn = 0 
            ssd = "-"
        pass
        self.ssr = ssr 
        self.ssn = ssn
        self.ssd = ssd 
        self.top = None
        


class GDMLSolidCheck(object):
    def __init__(self, gdml):
        self.gdml = gdml
        self.associate_solids_to_lv()
        self.analyse_solids()

    def associate_solids_to_lv(self):
        so2lv = {}
        lvs = self.gdml.volumes.values()
        for lv in lvs:
            solid = lv.solid
            so2lv[solid.idx] = lv.idx
            pass
        pass
        self.so2lv = so2lv

    def analyse_solids(self):
        """
        Builds tree of SNode in order to identify 
        top solids that are not part of any other solid.
        """
        flatsolids = self.gdml.solids.values()
        for solid in flatsolids:
            sn = SNode.create( solid.idx, solid.subsolids )
            ssr = solid.subsolidranges
            assert len(ssr) in (0,1), "all solids expected to be leaves with no subsolids or have contiguous idx range subsolids"
        pass
        tops = SNode.tops()
        ntops = len(tops)
        ndeep = 0 
        deeplv = []
   
        for top in tops:
            solid = self.gdml.solids(top.idx) 
            cn = solid.as_ncsg()
            cn.analyse() 

            lvidx = self.so2lv[solid.idx]
            lv = self.gdml.volumes(lvidx)

            if cn.height > 3: 
                ndeep += 1 
                print "solid.idx:%d cn.height:%d cn.totnodes:%d solid.name:%s ideep:%d lvidx:%d lvn:%s " % (solid.idx, cn.height, cn.totnodes, solid.name, ndeep, lvidx, lv.name  )
                deeplv.extend(["%2d : %-60s : %s " % (ndeep, lv.name,repr(cn))])
                print cn.txt
            pass
        pass
        log.info("analyse_solids nflatsolids:%d ntops:%d ndeep:%d " % (len(flatsolids), ntops, ndeep)) 
        
        print "\n".join(deeplv)




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
    nodelist = target.rprogeny(gmaxdepth, gmaxnode)
    log.info(" target nodelist  %s " % len(nodelist) )   


    gsc = GDMLSolidCheck(gdml)







