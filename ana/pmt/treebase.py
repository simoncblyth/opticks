#!/usr/bin/env python
"""
treebase.py
=============

* Node and Tree classes work together to provide tree "auto" assemblage 
  using parent digest lookup

* Stripped volume PV/LV/PV/LV source trees in detdesc or gdml flavors are converted 
  into a homogenous Node trees (PV,LV)/(PV,LV)/...


"""

import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 

from opticks.ana.base import opticks_main, Buf

log = logging.getLogger(__name__)


class DummyTopPV(object):
    name = "dummyTopPV"
    def find_(self, smth):
        return None
    def __repr__(self):
        return self.name


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
    def _depth(cls, node):
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1 
        pass
        return d 

    @classmethod
    def create(cls, volpath, lvtype="Logvol", pvtype="Physvol", postype="posXYZ" ):
        """
        :param volpath: ancestor volume instances
        """
        assert len(volpath) >= 2 
        
        node = cls(volpath) 

        ndig = node.digest   
        assert ndig not in Tree.registry, "each node must have a unique digest" 

        node.index  = len(Tree.registry)

        Tree.byindex[node.index] = node 
        Tree.registry[ndig] = node

        parent = Tree.lookup(node.pdigest)
        node.parent = parent

        if node.parent:
            node.parent.add_child(node)  
        pass
        node.depth = cls._depth(node)

        assert type(volpath[-2]).__name__ in (pvtype, "DummyTopPV") 
        assert type(volpath[-1]).__name__ == lvtype

        pv = volpath[-2] 
        lv = volpath[-1] 

        assert lv
  
        if node.index > 0:
            if pv is None:
                log.fatal("all nodes other than root must have a pv %r " % node)
            assert pv 
        pass

        node.pv = pv
        node.lv = lv


        node.posXYZ = node.pv.find_(postype) if node.pv is not None else None

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
        self.depth = None

    def visit(self, depth):
        log.info("visit depth %s %s " % (depth, repr(self)))

    def traverse(self, depth=0):
        self.visit(depth)
        for child in self.children:
            child.traverse(depth+1)

    def rprogeny(self, maxdepth=0, maxnode=0):
        """
        :return list of nodes:  
        """
        progeny = []
        skip = dict(total=0,count=0,depth=0)

        def progeny_r(node, depth):
            count_ok = maxnode == 0 or len(progeny) < maxnode
            depth_ok = maxdepth == 0 or depth < maxdepth 
            if count_ok and depth_ok:
                progeny.append(node) 
            else:
                skip["total"] += 1
                if not count_ok: skip["count"] += 1 
                if not depth_ok: skip["depth"] += 1 
            pass
            for child in node.children:
                progeny_r(child, depth+1)
            pass
        pass

        progeny_r(self, 0)
        pass
        log.info("rprogeny numProgeny:%s (maxnode:%s maxdepth:%s skip:%r ) " % (len(progeny),maxnode,maxdepth,skip ))
        return progeny

    def add_child(self, child):
        log.debug("add_child %s " % repr(child))
        self.children.append(child)

    def filter_children_by_lvn(self, lvn):
        return filter(lambda node:node.lv.name.startswith(lvn), self.children )

    siblings = property(lambda self:self.parent.filter_children_by_lvn(self.lv.name), doc="siblings of this node with same lv" )

    def dump(self, msg="Node.dump"):
        log.info(msg + " " + repr(self))
        #print "\n".join(map(str, self.geometry))   

    nchild = property(lambda self:len(self.children))
    name = property(lambda self:"Node %2d : dig %s pig %s depth %s nchild %s " % (self.index, self.digest[:4], self.pdigest[:4], self.depth, self.nchild) )

    def __repr__(self):
        return "%s \npv:%s\nlv:%s : %s " % (self.name, repr(self.pv),repr(self.lv), repr(self.posXYZ) ) 




class Tree(object):
    """
    Following the pattern used in assimpwrap-/AssimpTree 
    creates paired volume tree::

         (pv,lv)/(pv,lv)/ ...

    Which is more convenient to work with than the 
    striped volume tree obtained from the XML parse
    (with Elem wrapping) of form:

         pv/lv/pv/lv/.. 

    Note that the point of this is to create a tree at the 
    desired granularity (with nodes encompassing PV and LV)
    which can be serialized into primitives for analytic geometry ray tracing.

    """
    registry = {}
    byindex = {}

    @classmethod
    def clear(cls):
        cls.registry.clear()
        cls.byindex.clear()

    @classmethod
    def lookup(cls, digest):
        return cls.registry.get(digest, None)  

    @classmethod
    def get(cls, index):
        return cls.byindex.get(index, None)  

    @classmethod
    def filternodes_pv(cls, pfx):
        return filter(lambda node:node.pv.name.startswith(pfx), cls.byindex.values()) 

    @classmethod
    def filternodes_lv(cls, pfx):
        return filter(lambda node:node.lv.name.startswith(pfx), cls.byindex.values()) 

    @classmethod
    def filternodes_so(cls, pfx):
        """
        NB this only finds top level solids 
        ::
        
            t.filternodes_so("gds0xc28d3f0")  # works
            t.filternodes_so("gds_polycone0xc404f40")  # nope thats hidden in union

        """
        def filter_lv_solid(node):
            return node.lv.solid is not None and node.lv.solid.name.startswith(pfx)  
        pass
        return filter(filter_lv_solid, cls.byindex.values()) 


    @classmethod
    def findnode_lv(cls, lvn, idx=0):
        """
        TODO: use the idx within filternodes for short circuiting 
        """
        nodes = cls.filternodes_lv(lvn)    

        numNodes = len(nodes) 
        log.info("found %s nodes with lvn(LV name prefix) starting:%s " % (numNodes, lvn))
        if not idx < numNodes:
             log.warning("requested node index idx %s not within numNodes %s " % (idx, numNodes)) 
        pass

        targetNode = nodes[idx] if idx < numNodes else None 

        log.info("selected targetNode:[%s]\n%r " % (idx, targetNode)) 
        return targetNode

    @classmethod
    def findnode(cls, sel, idx=None):
        try:
            isel = int(sel)
        except ValueError:
            isel = None 
        pass

        if isel is not None:
            targetNode = cls.get(isel)
        else:
            targetNode = cls.findnode_lv(sel, idx)
        pass

        if targetNode is None:
            log.warning("failed to findnode with sel %s and idx %s " % (sel, idx))
            return None
        return targetNode


    @classmethod
    def subtree(cls, sel, maxdepth=0, maxnode=0, idx=0):
        """
        :param sel: lvn prefix or integer tree index
        :param maxdepth: node depth limit
        :param maxnode: node count limit
        :param idx: used to pick target node within an lvn selection that yields multiple nodes

        TODO: review similar node selection code from previous DAE based approach
        """
        targetNode = cls.findnode(sel, idx)
        pass
        progeny = targetNode.rprogeny(maxdepth, maxnode)   # including the idxNode 
        return progeny

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

    def __call__(self, arg):
        if type(arg) is int:
            return self.get(arg)
        elif type(arg) is slice:
            return [self.get(index) for index in range(arg.start, arg.stop, arg.step or 1 )]
        else:
             log.warning("expecting int or slice")
             return None


    def __getitem__(self, arg):
        slice_ = arg if type(arg) is slice else slice(arg,arg+1,1) 
        self.slice_ = slice_ 
        return self

    typ = property(lambda self:self.__class__.__name__)

    def __repr__(self):
        nn = self.num_nodes()
        smry  =  "%s num_nodes %s " % (self.typ, nn)
        lines = [smry]
        s = self.slice_
        if s is not None:
            step = s.step if s.step is not None else 1
            lines.extend(map(lambda index:repr(self.byindex[index]), range(s.start,s.stop,step)))
        pass
        return "\n\n".join(lines)


    def traverse(self):
        self.root.traverse()

    def __init__(self, base):
        """
        :param base: top ddbase.Elem or gdml.G instance of lv of interest, eg lvPmtHemi
        """
        self.clear()  # prevent interactive re-running from doubling up nodes

        self.lvtype = base.lvtype
        self.pvtype = base.pvtype
        self.postype = base.postype
        assert self.postype and self.lvtype and self.pvtype

        self.slice_ = None

        self.base = base

        top = DummyTopPV()
        ancestors = [top] # dummy to regularize striping TOP-LV-PV-LV 
        self.root = self.create_r(self.base, ancestors)

    def create_r(self, vol, ancestors):
        """
        Source tree traversal creating nodes as desired in destination tree

        #. vital to make a copy with [:] as need separate volpath for every node

        #. only form destination nodes at Logvol points in the tree

        #. this is kept simple as the parent digest approach to tree hookup
           means that the Nodes assemble themselves into the tree, just need
           to create nodes where desired and make sure to traverse the entire 
           source tree
        """
        volpath = ancestors[:] 
        volpath.append(vol) 

        node = None
        if type(volpath[-1]).__name__ == self.lvtype:
            node = Node.create(volpath, lvtype=self.lvtype, pvtype=self.pvtype, postype=self.postype )
        pass

        for child in vol.children:
            self.create_r(child, volpath)
        pass 
        return node





if __name__ == '__main__':

    args = opticks_main()

    from ddbase import Dddb

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)





