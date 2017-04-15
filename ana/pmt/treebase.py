#!/usr/bin/env python
"""
treebase.py
=============

treebase.py uses self-assemblage by digest approach, it contains
a tree of Node instances starting from root

::

    In [1]: run treebase.py
    ...
    In [2]: tr.root
    Out[2]: Node  0 : dig 8c5f pig d41d : LV lvPmtHemi                           Pyrex None : None  : None 

    In [5]: tr.byindex
    Out[5]: 
    {0: Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None ,
     1: Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None ,
     2: Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None ,
     3: Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0    ,
     4: Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5    }

    In [6]: tr.registry
    Out[6]: 
    {'324d9022d803eae989b540bbb2375f38': Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None ,
     '5e291f5b9bbedf27f720e5dceb65ad56': Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5    ,
     '9e612c43301f13a23a5fd41e7ea59404': Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0    ,
     'f34ba27750136ebdc5bd9f3119f2c559': Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None ,
     'fafaa4fcd3682ac3f89da9afb9680e9a': Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None }

    In [7]: 


Recursive dumper::

    In [20]: tr.get(0).traverse()
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  0 : dig f34b pig d41d : LV lvPmtHemi                           Pyrex None : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     
    [2017-04-14 19:51:48,181] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 2 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     

    In [21]: tr.get(1).traverse()
    [2017-04-14 19:52:01,124] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  1 : dig fafa pig f34b : LV lvPmtHemiVacuum                    Vacuum None : None  : None  
    [2017-04-14 19:52:01,124] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  
    [2017-04-14 19:52:01,125] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     
    [2017-04-14 19:52:01,125] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 1 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     

    In [22]: tr.get(2).traverse()
    [2017-04-14 19:52:17,660] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  2 : dig 324d pig fafa : LV lvPmtHemiCathode                 Bialkali DsPmtSensDet : None  : None  

    In [23]: tr.get(3).traverse()
    [2017-04-14 19:52:29,365] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  3 : dig 9e61 pig fafa : LV lvPmtHemiBottom              OpaqueVacuum None : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     : PosXYZ  PmtHemiFaceOff+PmtHemiBellyOff : 69.0     

    In [24]: tr.get(4).traverse()
    [2017-04-14 19:52:42,476] p36676 {/Users/blyth/opticks/ana/pmt/tree.py:79} INFO - visit depth 0 Node  4 : dig 5e29 pig fafa : LV lvPmtHemiDynode              OpaqueVacuum None : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     : PosXYZ  -0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness : -81.5     


::

     37   <!-- The PMT glass -->
     38   <logvol name="lvPmtHemi" material="Pyrex">
     39     <union name="pmt-hemi">
     40       <intersection name="pmt-hemi-glass-bulb">
     41     <sphere name="pmt-hemi-face-glass"
     42         outerRadius="PmtHemiFaceROC"/>
     43 
     44     <sphere name="pmt-hemi-top-glass"
     45         outerRadius="PmtHemiBellyROC"/>
     46     <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     47 
     48     <sphere name="pmt-hemi-bot-glass"
     49         outerRadius="PmtHemiBellyROC"/>
     50     <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     51 
     52       </intersection>
     53       <tubs name="pmt-hemi-base"
     54         sizeZ="PmtHemiGlassBaseLength"
     55         outerRadius="PmtHemiGlassBaseRadius"/>
     56       <posXYZ z="-0.5*PmtHemiGlassBaseLength"/>
     57     </union>
     58 
     59     <physvol name="pvPmtHemiVacuum"
     60          logvol="/dd/Geometry/PMT/lvPmtHemiVacuum"/>
     61 
     62   </logvol>

::

    In [48]: py = tr.get(0)

    In [54]: py.lv.name
    Out[54]: 'lvPmtHemi'

    In [55]: py.lv.material
    Out[55]: 'Pyrex'

    In [57]: py.lv.findall_("./*")
    Out[57]: 
    [Union             pmt-hemi  ,
     PV pvPmtHemiVacuum      /dd/Geometry/PMT/lvPmtHemiVacuum ]

    In [58]: un = py.lv.findall_("./*")[0]

    In [59]: un
    Out[59]: Union             pmt-hemi  

    In [60]: un.findall_("./*")
    Out[60]: 
    [Intersection  pmt-hemi-glass-bulb  ,
     Tubs        pmt-hemi-base : outerRadius PmtHemiGlassBaseRadius : 42.25   sizeZ PmtHemiGlassBaseLength : 169.0   :  None ,
     PosXYZ  -0.5*PmtHemiGlassBaseLength : -84.5   ]


Need to findall_ recurse on the lv, constructing NCSG node tree.
Unclear what level to do this at, probably simpler to operate at dd level


"""

import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 

from opticks.ana.base import opticks_main, Buf

from ddbase import Dddb


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
        :param volpath: list of python instance id, effectively reference addresses

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

    def traverse(self):
        self.root.traverse()

    def __init__(self, base):
        """
        :param base: top ddbase.Elem instance of lv of interest, eg lvPmtHemi
        """
        self.base = base
        ancestors = [self]   # dummy top "PV", to regularize striping: TOP-LV-PV-LV 
        self.root = self.traverseWrap_(self.base, ancestors)

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

    args = opticks_main()

    g = Dddb.parse(args.apmtddpath)

    lv = g.logvol_("lvPmtHemi")

    tr = Tree(lv)





