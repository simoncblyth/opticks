#!/usr/bin/env python
"""

CSG_
=====

Used by tboolean- for CSG solid construction and serialization
for reading by npy-/NCSG 

TODO:

* collect metadata kv for any node, for root nodes only 
  persist the metadata with Serialize (presumably as json or ini) 
  For example to control type of triangulation and 
  parameters such as thresholds and octree sizes per csg tree.

"""
import os, sys, logging, json, numpy as np
log = logging.getLogger(__name__)

# bring in enum values from sysrap/OpticksCSG.h
from opticks.sysrap.OpticksCSG import CSG_

Q0,Q1,Q2,Q3 = 0,1,2,3
X,Y,Z,W = 0,1,2,3

TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )
TREE_EXPECTED = map(TREE_NODES, range(10))   # [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]


class CSG(CSG_):
    """
    Serialization layout here must echo that in NCSG 
    """
    NJ, NK = 4, 4
    FILENAME = "csg.txt"

    def depth_(self):
        def depth_r(node, depth):
            node.depth = depth
            if node.left is None and node.right is None:
                return depth
            else:
                ldepth = depth_r(node.left, depth+1)
                rdepth = depth_r(node.right, depth+1)
                return max(ldepth, rdepth)
            pass
        pass
        return depth_r(self, 0) 

    def analyse(self):
        self.height = self.depth_()
        self.totnodes = TREE_NODES(self.height)

    is_root = property(lambda self:hasattr(self,'height') and hasattr(self,'totnodes'))

    @classmethod
    def treedir(cls, base, idx):
        return os.path.join(base, "%d" % idx )

    @classmethod
    def txtpath(cls, base):
        return os.path.join(base, cls.FILENAME )

    @classmethod
    def Serialize(cls, trees, base):
        assert type(trees) is list 
        assert type(base) is str and len(base) > 5, ("invalid base directory %s " % base)
        base = os.path.expandvars(base) 
        log.info("CSG.Serialize : writing %d trees to directory %s " % (len(trees), base))
        if not os.path.exists(base):
            os.makedirs(base)
        pass
        for it, tree in enumerate(trees):
            treedir = cls.treedir(base,it)
            if not os.path.exists(treedir):
                os.makedirs(treedir)
            pass
            tree.save(treedir)
        pass
        boundaries = map(lambda tree:tree.boundary, trees)
        open(cls.txtpath(base),"w").write("\n".join(boundaries))
        print base # used from bash to pass directory of serialization into testconfig

    @classmethod
    def Deserialize(cls, base):
        base = os.path.expandvars(base) 
        assert os.path.exists(base)
        boundaries = file(cls.txtpath(base)).read().splitlines()
        trees = []
        for idx, boundary in enumerate(boundaries): 
            tree = cls.load(cls.treedir(base, idx))      
            tree.boundary = boundary 
            trees.append(tree)
        pass
        return trees

    @classmethod
    def make_transform(cls, translate, rotate):
        if translate is None and rotate is None: return None
        tla_  = lambda s:np.fromstring(s, dtype=np.float32, sep=",") if s is not None else np.zeros(3, dtype=np.float32)
        rot_  = lambda s:np.fromstring(s, dtype=np.float32, sep=",") if s is not None else np.eye(3, dtype=np.float32)
        tran = np.eye(4, dtype=np.float32)
        tran[:3, :3] = rot_(rotate)
        tran[3,:3] = tla_(translate)
        return tran
 


    def serialize(self):
        """
        Array is sized for a complete tree, empty slots stay all zero

        For transforms, need to 

        """
        if not self.is_root: self.analyse()
        buf = np.zeros((self.totnodes,self.NJ,self.NK), dtype=np.float32 )

        transforms = []

        def serialize_r(node, idx): 

            tran = node.rtransform  
            if tran is None:
                itra = 0 
            else:
                transforms.append(tran)
                itra = len(transforms)   # 1-based index pointing to the transform
            pass

            buf[idx] = node.asarray(itra)

            if node.left is not None and node.right is not None:
                serialize_r( node.left,  2*idx+1)
                serialize_r( node.right, 2*idx+2)
            pass
        serialize_r(self, 0)

        tbuf = np.vstack(transforms).reshape(-1,4,4) if len(transforms) > 0 else None 
        return buf, tbuf


    def save(self, treedir):
        """
        """
        nodepath = self.nodepath(treedir)
        metapath = self.metapath(treedir)
        tranpath = self.tranpath(treedir)

        log.info("save to %s meta %r metapath %s tpath %s " % (nodepath, self.meta, metapath, tranpath))
        json.dump(self.meta,file(metapath,"w"))

        nodebuf, tranbuf = self.serialize() 

        np.save(nodepath, nodebuf)

        if tranbuf is not None:
            np.save(tranpath, tranbuf)
        pass

    stream = property(lambda self:self.save(sys.stdout))

    @classmethod
    def tranpath(cls, treedir):
        return os.path.join(treedir,"transforms.npy") 
    @classmethod
    def metapath(cls, treedir):
        return os.path.join(treedir,"meta.json") 
    @classmethod
    def nodepath(cls, treedir):
        return os.path.join(treedir,"nodes.npy") 


    @classmethod
    def load(cls, treedir):
        tree = cls.deserialize(treedir) 
        log.info("load %s DONE -> %r " % (treedir, tree) )
        return tree

    @classmethod
    def deserialize(cls, treedir):
        assert os.path.exists(treedir)
         
        nodepath = cls.nodepath(treedir)
        metapath = cls.metapath(treedir)
        tranpath = cls.tranpath(treedir)

        log.info("load nodepath %s tranpath %s " % (nodepath,tranpath) )

        nodebuf = np.load(nodepath) 
        tranbuf = np.load(tranpath) if os.path.exists(tranpath) else None

        totnodes = len(nodebuf)
        try:
            height = TREE_EXPECTED.index(totnodes)
        except ValueError:
            log.fatal("invalid serialization of length %d not in expected %r " % (totnodes,TREE_EXPECTED))
            assert 0

        def deserialize_r(buf, idx):
            node = cls.fromarray(buf[idx]) if idx < len(buf) else None
            if node is not None and node.itra is not None and node.itra > 0:
                assert tranbuf is not None and node.itra - 1 < len(tranbuf)  
                node.rtransform = tranbuf[node.itra - 1]
                
            if node is not None:
                node.left  = deserialize_r(buf, 2*idx+1)
                node.right = deserialize_r(buf, 2*idx+2)
            pass
            return node  
        pass
        root = deserialize_r(nodebuf, 0)
        root.totnodes = totnodes
        root.height = height 
        return root


    

    def __init__(self, typ, left=None, right=None, param=None, boundary="", rtranslate=None, rrotate=None,  **kwa):
        if type(typ) is str:
            typ = self.fromdesc(typ)  
        pass
        assert type(typ) is int and typ > -1, (typ, type(typ))

        self.typ = typ
        self.left = left
        self.right = right
        self.param = param
        self.boundary = boundary
        self.rtransform = self.make_transform(rtranslate,rrotate)
        self.meta = kwa

    def _get_param(self):
        return self._param
    def _set_param(self, v):
        self._param = np.asarray(v) if v is not None else None
    param = property(_get_param, _set_param)

    def asarray(self, itra=0):
        arr = np.zeros( (self.NJ, self.NK), dtype=np.float32 )
        
        if self.param is not None:   # avoid gibberish in buffer
            arr[Q0] = self.param
            assert self.rtransform == None
        elif self.rtransform is not None:
            assert itra > 0, itra  # 1-based transform index
            arr.view(np.uint32)[Q3,W] = itra 
        else:
            pass
        pass
        arr.view(np.uint32)[Q2,W] = self.typ

        return arr

    @classmethod
    def fromarray(cls, arr):
        typ = int(arr.view(np.uint32)[Q2,W])
        itra = int(arr.view(np.uint32)[Q3,W]) if typ < cls.SPHERE else 0 

        log.info("CSG.fromarray typ %d %s itra %d  " % (typ, cls.desc(typ), itra) )
        n = cls(typ) if typ > 0 else None
        if n is not None:
            n.itra = itra if itra > 0 else None
        pass
        return n 


    def __repr__(self):
        rrep = "height:%d totnodes:%d " % (self.height, self.totnodes) if self.is_root else ""  
        return "%s(%s) %s " % (self.desc(self.typ), ",".join(map(repr,filter(None,[self.left,self.right]))),rrep)

    def __call__(self, p):
        """
        SDF : signed distance field
        """
        if self.typ == self.SPHERE:
            center = self.param[:3]
            radius = self.param[3]
            pc = np.asarray(p) - center
            return np.sqrt(np.sum(pc*pc)) - radius 
        else:
            assert 0 

    is_primitive = property(lambda self:self.typ >= self.SPHERE )

    def union(self, other):
        return CSG(typ=self.UNION, left=self, right=other)

    def subtract(self, other):
        return CSG(typ=self.DIFFERENCE, left=self, right=other)

    def intersect(self, other):
        return CSG(typ=self.INTERSECTION, left=self, right=other)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, other):
        return self.intersect(other)

       


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum" )
   
    s = CSG("sphere")
    b = CSG("box")
    sub = CSG("union", left=s, right=b, rtranslate="0,0,20", boundary="Vacuum///GlassShottF2", hello="world")

    trees0 = [container, sub]


    base = "$TMP/csg_py"
    CSG.Serialize(trees0, base )
    trees1 = CSG.Deserialize(base)

    assert len(trees1) == len(trees0)

    for i in range(len(trees1)):
        assert np.all( trees0[i].rtransform == trees1[i].rtransform )



   

    
