#!/usr/bin/env python

import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 


from opticks.ana.base import opticks_main, Buf, manual_mixin

from ddbase import Dddb
from ddpart import Parts, ddpart_manual_mixin

from geom import Part

log = logging.getLogger(__name__)


from opticks.analytic.treebase import Tree, Node


class NodePartitioner(object):
    """
    All NodePartitioner methods are added to treebase.Node 
    on calling the below treepart_manual_mixin function 
    """ 
    def parts(self):
        """
        Divvy up geometry into parts that 
        split "intersection" into union lists. This boils
        down to judicious choice of bounding box according 
        to intersects of the source gemetry.
        """
        if not hasattr(self, '_parts'):
            _parts = self.lv.parts()
            for p in _parts:
                p.node = self
            pass
            self._parts = _parts 
        pass
        return self._parts

    def num_parts(self):
        parts = self.parts()
        return len(parts)


class TreePartitioner(object):
    """
    All TreePartitioner methods are added to treebase.Tree
    on calling the below treepart_manual_mixin function 
    """ 
    @classmethod
    def num_parts(cls):
        nn = cls.num_nodes()
        tot = 0 
        for i in range(nn):
            node = cls.get(i)
            tot += node.num_parts()
        pass
        return tot

    @classmethod
    def parts(cls):
        tnodes = cls.num_nodes() 
        tparts = cls.num_parts() 
        log.info("tnodes %s tparts %s " % (tnodes, tparts))

        pts = Parts()
        gcsg = []

        for i in range(tnodes):
            node = cls.get(i)

            log.debug("tree.parts node %s parent %s" % (repr(node),repr(node.parent)))
            log.info("tree.parts node.lv %s " % (repr(node.lv)))
            log.info("tree.parts node.pv %s " % (repr(node.pv)))

            npts = node.parts()
            pts.extend(npts)    

            if hasattr(npts, 'gcsg') and len(npts.gcsg) > 0:
                for c in npts.gcsg:
                    c.node = node
                pass
                gcsg.extend(npts.gcsg)  
            pass
        pass
        assert len(pts) == tparts          
        pts.gcsg = gcsg 
        return pts 

    @classmethod
    def convert(cls, parts, explode=0.):
        """
        :param parts: array of parts
        :return: np.array buffer of parts

        Tree.convert

        #. collect Part instances from each of the nodes into list
        #. serialize parts into array, converting relationships into indices
        #. this cannot live at lower level as serialization demands to 
           allocate all at once and fill in the content, also conversion
           of relationships to indices demands an all at once conversion

        Five solids of DYB PMT represented in part buffer

        * part.typecode 1:sphere, 2:tubs
        * part.flags, only 1 for tubs
        * part.node.index 0,1,2,3,4  (0:4pt,1:4pt,2:2pt,3:1pt,4:1pt) 

        ::

            In [19]: p.buf.view(np.int32)[:,(1,2,3),3]
            Out[19]: 
              Buf([[0, 1, 0],       part.flags, part.typecode, nodeindex    
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 2, 0],

                   [0, 1, 1],
                   [0, 1, 1],
                   [0, 1, 1],
                   [1, 2, 1],

                   [0, 1, 2],
                   [0, 1, 2],

                   [0, 1, 3],

                   [0, 2, 4]], dtype=int32)


            In [22]: p.buf.view(np.int32)[:,1,1]     # 1-based part index
            Out[22]: Buf([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)


        * where are the typecodes hailing from, not using OpticksCSG.h enum ?
          nope hardcoded into geom.py Part.__init__  Sphere:1, Tubs:2 Box:3

      

        """
        data = np.zeros([len(parts),4,4],dtype=np.float32)
        for i,part in enumerate(parts):
            #print "part (%d) tc %d  %r " % (i, part.typecode, part)
            data[i] = part.as_quads()

            data[i].view(np.int32)[1,1] = i + 1           # 1-based part index, where parent 0 means None
            data[i].view(np.int32)[1,2] = 0               # set to boundary index in C++ ggeo-/GPmt
            data[i].view(np.int32)[1,3] = part.flags      # used in intersect_ztubs
            data[i].view(np.int32)[2,3] = part.typecode   # bbmin.w : typecode 
            data[i].view(np.int32)[3,3] = part.node.index # bbmax.w : solid index  

            if explode>0:
                dx = i*explode
                data[i][0,0] += dx
                data[i][2,0] += dx
                data[i][3,0] += dx
            pass
        pass
        buf = data.view(Buf) 
        buf.boundaries = map(lambda _:_.boundary, parts) 

        if hasattr(parts, "gcsg"):
            buf.gcsg = parts.gcsg 
            buf.materials = map(lambda cn:cn.lv.material,filter(lambda cn:cn.lv is not None, buf.gcsg))
            buf.lvnames = map(lambda cn:cn.lv.name,filter(lambda cn:cn.lv is not None, buf.gcsg))
            buf.pvnames = map(lambda lvn:lvn.replace('lv','pv'), buf.lvnames)
        pass
        return buf



def treepart_manual_mixin():
    """
    Using manual mixin approach to avoid changing 
    the class hierarchy whilst still splitting base
    functionality from partitioner methods.  
    """
    manual_mixin(Node, NodePartitioner)
    manual_mixin(Tree, TreePartitioner)



if __name__ == '__main__':

    args = opticks_main()

    ddpart_manual_mixin()  # add methods to Tubs, Sphere, Elem and Primitive
    treepart_manual_mixin() # add methods to Node and Tree

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    partsbuf = tr.convert(parts) 

    log.warning("use analytic so save the PMT, this is just for testing tree conversion")


