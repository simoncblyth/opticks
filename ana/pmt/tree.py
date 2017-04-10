#!/usr/bin/env python
"""

"""
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 


from opticks.ana.base import Buf
from dd import Dddb, Parts, Union, Intersection 
from csg import CSG
from geom import Part


log = logging.getLogger(__name__)

class Node(object):
    @classmethod
    def md5digest(cls, volpath ):
        """  
        Use of id means that will change from run to run. 
        """
        dig = ",".join(map(lambda _:str(id(_)),volpath))
        dig = hashlib.md5(dig).hexdigest() 
        return dig

    @classmethod
    def create(cls, volpath ):
        """
        Note that this parent digest approach allows the 
        nodes to assemble themselves into the tree  
        """
        assert len(volpath) >= 2 
        
        node = cls(volpath) 

        ndig = node.digest   
        assert ndig not in Tree.registry, "each node must have a unique digest" 
        node.index  = len(Tree.registry)

        Tree.byindex[node.index] = node 
        Tree.registry[ndig] = node

        node.parent = Tree.lookup(node.pdigest)
        if node.parent:
            node.parent.add_child(node)  

        node.pv = volpath[-2] if type(volpath[-2]).__name__ == "Physvol" else None  # tis None for root
        node.lv = volpath[-1] if type(volpath[-1]).__name__ == "Logvol" else None
        assert node.lv

        node.posXYZ = node.pv.find_("./posXYZ") if node.pv is not None else None

        #node.dump("visitWrap_")
        return node

    def __init__(self, volpath):
        """
        :param volpath: list of volume instances thru the volume tree

        Each node in the derived tree corresponds to two levels of the 
        source XML nodes tree, ie the lv and pv.
        So pdigest from backing up two levels gives access to parent node.
        """
        self.volpath = volpath
        self.digest = self.md5digest( volpath[0:len(volpath)] )
        self.pdigest = self.md5digest( volpath[0:len(volpath)-2] )

        # Node constituents are set by Tree
        self.parent = None
        self.index = None
        self.posXYZ = None
        self.children = []
        self.lv = None
        self.pv = None
        self._parts = None

    def visit(self, depth):
        log.info("visit depth %s %s " % (depth, repr(self)))

    def traverse(self, depth=0):
        self.visit(depth)
        for child in self.children:
            child.traverse(depth+1)

    def add_child(self, child):
        log.debug("add_child %s " % repr(child))
        self.children.append(child)

    def dump(self, msg="Node.dump"):
        log.info(msg + " " + repr(self))
        #print "\n".join(map(str, self.geometry))   

    def __repr__(self):
        return "Node %2d : dig %s pig %s : %s : %s " % (self.index, self.digest[:4], self.pdigest[:4], repr(self.volpath[-1]), repr(self.posXYZ) ) 


    def parts(self):
        """
        Divvy up geometry into parts that 
        split "intersection" into union lists. This boils
        down to judicious choice of bounding box according 
        to intersects of the source gemetry.
        """
        if self._parts is None:
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



class Tree(object):
    """
    Following pattern of assimpwrap-/AssimpTree 
    transforming tree from  pv/lv/pv/lv/.. to   (pv,lv)/(pv,lv)/ ...

    Note that the point of this is to create a tree at the 
    desired granularity (with nodes encompassing PV and LV)
    which can be serialized into primitives for analytic geometry ray tracing.
    """
    registry = {}
    byindex = {}

    @classmethod
    def lookup(cls, digest):
        return cls.registry.get(digest, None)  

    @classmethod
    def get(cls, index):
        return cls.byindex.get(index, None)  

    @classmethod
    def description(cls):
        return "\n".join(["%s : %s " % (k,v) for k,v in cls.byindex.items()])

    @classmethod
    def dump(cls):
        print cls.description()

    @classmethod
    def num_nodes(cls):
        assert len(cls.registry) == len(cls.byindex)
        return len(cls.registry)

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
        csg = []

        for i in range(tnodes):
            node = cls.get(i)
            log.debug("tree.parts node %s parent %s" % (repr(node),repr(node.parent)))
            log.info("tree.parts node.lv %s " % (repr(node.lv)))
            log.info("tree.parts node.pv %s " % (repr(node.pv)))
            npts = node.parts()
            #print npts
            pts.extend(npts)    

            if hasattr(npts, 'csg') and len(npts.csg) > 0:
                for c in npts.csg:
                    c.node = node
                csg.extend(npts.csg)  

        pass
        assert len(pts) == tparts          
        pts.csg = csg 
        return pts 

    @classmethod
    def convert(cls, parts, explode=0., analytic_version=0):
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
            data[i] = part.as_quads(analytic_version=analytic_version)

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

        if hasattr(parts, "csg"):
            buf.csg = parts.csg 
            buf.materials = map(lambda cn:cn.lv.material,filter(lambda cn:cn.lv is not None, buf.csg))
            buf.lvnames = map(lambda cn:cn.lv.name,filter(lambda cn:cn.lv is not None, buf.csg))
            buf.pvnames = map(lambda lvn:lvn.replace('lv','pv'), buf.lvnames)
        pass
        return buf


    @classmethod
    def save(cls, path_, buf):
        assert 0, "moved to GPmt.save" 

    def traverse(self):
        self.wrap.traverse()

    def __init__(self, base):
        """
        :param base: top dd.Elem instance of lv of interest, eg lvPmtHemi
        """
        self.base = base
        ancestors = [self]   # dummy top "PV", to regularize striping: TOP-LV-PV-LV 
        self.wrap = self.traverseWrap_(self.base, ancestors)

    def traverseWrap_(self, vol, ancestors):
        """
        Source tree traversal, creating nodes as desired in destination tree

        #. vital to make a copy with [:] as need separate volpath for every node
        #. only form wrapped nodes at Logvol points in the tree
           in order to have regular TOP-LV-PV-LV ancestry, 
           but traverse over all nodes of the source tree
        #. this is kept simple as the parent digest approach to tree hookup
           means that the Nodes assemble themselves into the tree, just need
           to create nodes where desired and make sure to traverse the entire 
           source tree
        """
        volpath = ancestors[:] 
        volpath.append(vol) 

        ret = None
        if type(volpath[-1]).__name__ == "Logvol":
            ret = self.visitWrap_(volpath)

        for child in vol.children():
            self.traverseWrap_(child, volpath)
        pass 
        return ret

    def visitWrap_(self, volpath):
        log.debug("visitWrap_ %s : %s " % (len(volpath), repr(volpath[-1])))
        return Node.create(volpath)







if __name__ == '__main__':
    format_ = "[%(filename)s +%(lineno)3s %(funcName)20s ] %(message)s" 
    logging.basicConfig(level=logging.INFO, format=format_)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)

    parts = tr.parts()

    partsbuf = tr.convert(parts) 

    log.warning("use analytic so save the PMT, this is just for testing tree conversion")


