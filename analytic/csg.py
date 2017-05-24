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
from opticks.analytic.glm import make_trs
from opticks.analytic.textgrid import TextGrid

Q0,Q1,Q2,Q3 = 0,1,2,3
X,Y,Z,W = 0,1,2,3

TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )
TREE_PRIMITIVES = lambda height:1 << height 
TREE_EXPECTED = map(TREE_NODES, range(10))   # [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]

fromstring_  = lambda s:np.fromstring(s, dtype=np.float32, sep=",")




class TreeBuilder(object):
    def __init__(self, primitives, operator="union"):
        """
        :param primitives: list of CSG instance primitives
        """
        self.operator = operator 
        nprim = len(map(lambda n:n.is_primitive, primitives))
        assert nprim == len(primitives) and nprim > 0

        self.nprim = nprim
        self.primitives = primitives

        height = -1
        for h in range(10):
            tprim = 1 << h 
            if tprim >= nprim:
                height = h
                break

        assert height > -1 
        log.debug("TreeBuilder nprim:%d required height:%d " % (nprim, height))
        self.height = height

        if height == 0:
            assert len(primitives) == 1
            root = primitives[0]
        else: 
            root = self.build(height)
            self.populate(root, primitives[::-1]) 
            self.prune(root)
        pass
        self.root = root 

    def build(self, height):
        """
        Build complete binary tree with all operators the same
        and CSG.ZERO placeholders for the primitives at elevation 0.
        """
        def build_r(elevation, operator):
            if elevation > 1:
                node = CSG(operator, name="treebuilder_midop")
                node.left  = build_r(elevation-1, operator) 
                node.right = build_r(elevation-1, operator)
            else:
                node = CSG(operator, name="treebuilder_bileaf")
                node.left = CSG(CSG.ZERO, name="treebuilder_bileaf_left")
                node.right = CSG(CSG.ZERO, name="treebuilder_bilead_right")
            pass
            return node  
        pass
        root = build_r(height, self.operator ) 
        return root

    def populate(self, root, primitives):
        """
        Replace the CSG.ZERO placeholders with the primitives
        """
        for node in root.inorder_():
            if node.is_operator:
                try:
                    if node.left.is_zero:
                        node.left = primitives.pop()
                    if node.right.is_zero:
                        node.right = primitives.pop()
                    pass
                except IndexError:
                    pass
                pass
            pass
        pass

    def prune(self, root):
        def prune_r(node):
            if node is None:return
            if node.is_operator:
                prune_r(node.left)
                prune_r(node.right)

                if node.left.is_lrzero:
                    node.left = CSG(CSG.ZERO)
                elif node.left.is_rzero:
                    ll = node.left.left
                    node.left = ll
                pass
                if node.right.is_lrzero:
                    node.right = CSG(CSG.ZERO)
                elif node.right.is_rzero:
                    rl = node.right.left
                    node.right = rl
                pass
            pass
        pass
        prune_r(root)





class CSG(CSG_):
    """
    Serialization layout here must echo that in NCSG 
    """
    NJ, NK = 4, 4
    FILENAME = "csg.txt"
    CONVEX_POLYHEDRA = [CSG_.TRAPEZOID]

    def depth_(self):
        """Label tree nodes with their depth, return maxdepth"""
        def depth_r(node, depth):
            if node is None:return depth
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

    def inorder_(self):
        """Return inorder sequence of nodes"""
        inorder = []
        def inorder_r(node):
            if node is None:return
            inorder_r(node.left)
            inorder.append(node)
            inorder_r(node.right)
        pass
        inorder_r(self)
        return inorder

    def parenting_(self):
        """Label tree nodes with their parent"""
        def parenting_r(node, parent):
            if node is None:return 
            node.parent = parent
            parenting_r(node.left, node)
            parenting_r(node.right, node)
        pass
        parenting_r(self, None)

    def rooting_(self):
        """Label tree nodes with root"""
        def rooting_r(node, root):
            if node is None:return 
            node.root = root
            rooting_r(node.left, root)
            rooting_r(node.right, root)
        pass
        rooting_r(self, self) 

    elevation = property(lambda self:self.root.height - self.depth) 

    def analyse(self):
        """
        Call from root node to label the tree, root node is annotated with:

        height 
             maximum node depth
 
        totnodes 
             complete binary tree node count for the height 
            
        ::

            In [45]: map(TREE_NODES, range(10))
            Out[45]: [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023] 

        """
        self.height = self.depth_()
        self.parenting_()
        self.rooting_()

        self.totnodes = TREE_NODES(self.height)
        inorder = self.inorder_()

        txt = TextGrid( self.height+1, len(inorder) )
        for i,node in enumerate(inorder):
            label = "%s%s" % (node.dsc, node.textra) if hasattr(node,'textra') else node.dsc
            txt.a[node.depth, i] = label 
        pass
        self.txt = txt


    is_root = property(lambda self:hasattr(self,'height') and hasattr(self,'totnodes'))

    @classmethod
    def treedir(cls, base, idx):
        return os.path.join(base, "%d" % idx )

    @classmethod
    def txtpath(cls, base):
        return os.path.join(base, cls.FILENAME )

    @classmethod
    def Serialize(cls, trees, base, outmeta=True):
        """
        :param trees: list of CSG instances of solid root nodes
        :param base: directory to save the tree serializations, under an indexed directory 
        """
        assert type(trees) is list 
        assert type(base) is str and len(base) > 5, ("invalid base directory %s " % base)
        base = os.path.expandvars(base) 
        log.info("CSG.Serialize : writing %d trees to directory %s " % (len(trees), base))
        if not os.path.exists(base):
            os.makedirs(base)
        pass
        for it, tree in enumerate(trees):
            treedir = cls.treedir(base,it)
            tree.save(treedir)
        pass

        boundaries = map(lambda tree:tree.boundary, trees)
        cls.CheckNonBlank(boundaries)
        open(cls.txtpath(base),"w").write("\n".join(boundaries))

        if outmeta:
            meta = dict(mode="PyCsgInBox", name=os.path.basename(base), analytic=1, csgpath=base)
            meta_fmt_ = lambda meta:"_".join(["%s=%s" % kv for kv in meta.items()])
            print meta_fmt_(meta)  # communicates to tboolean--
        pass

    @classmethod
    def CheckNonBlank(cls, boundaries):
        boundaries2 = filter(None, boundaries)
        assert len(boundaries) == len(boundaries2), "there are blank boundaries\n%s" % "\n".join(boundaries) 

    @classmethod
    def Deserialize(cls, base):
        base = os.path.expandvars(base) 
        assert os.path.exists(base)
        boundaries = file(cls.txtpath(base)).read().splitlines()
        cls.CheckNonBlank(boundaries)
        trees = []
        for idx, boundary in enumerate(boundaries): 
            tree = cls.load(cls.treedir(base, idx))      
            tree.boundary = boundary 
            trees.append(tree)
        pass
        return trees

    @classmethod
    def CubePlanes(cls, hsize=0.5):
        planes = np.zeros([6,4], dtype=np.float32)  # unit cube for testing
        for i in range(3):
            plp = np.array([int(i==0),int(i==1),int(i==2),hsize], dtype=np.float32)  
            pln = np.array([-int(i==0),-int(i==1),-int(i==2),hsize], dtype=np.float32)  
            planes[2*i+0] = plp
            planes[2*i+1] = pln
        pass
        return planes


    def serialize(self, suppress_identity=False):
        """
        Array is sized for a complete tree, empty slots stay all zero
        """
        if not self.is_root: self.analyse()
        buf = np.zeros((self.totnodes,self.NJ,self.NK), dtype=np.float32 )

        transforms = []
        planes = []

        def serialize_r(node, idx): 
            """
            :param node:
            :param idx: 0-based complete binary tree index, left:2*idx+1, right:2*idx+2 
            """
            trs = node.transform  
            if trs is None and suppress_identity == False:
                trs = np.eye(4, dtype=np.float32)  
                # make sure root node always has a transform, incase of global placement 
                # hmm root node is just an op-node it doesnt matter, need transform slots for all primitives 
            pass

            if trs is None:
                itransform = 0 
            else:
                itransform = len(transforms) + 1  # 1-based index pointing to the transform
                transforms.append(trs)
            pass


            node_planes = node.planes
            if len(node_planes) == 0:
                planeIdx = 0
                planeNum = 0
            else:
                planeIdx = len(planes) + 1   # 1-based index pointing to the first plane for the node
                planeNum = len(node_planes)
                planes.extend(node_planes)
            pass 
            log.debug("serialize_r idx %3d itransform %2d planeIdx %2d " % (idx, itransform, planeIdx))

            buf[idx] = node.as_array(itransform, planeIdx, planeNum)

            if node.left is not None and node.right is not None:
                serialize_r( node.left,  2*idx+1)
                serialize_r( node.right, 2*idx+2)
            pass
        pass

        serialize_r(self, 0)

        tbuf = np.vstack(transforms).reshape(-1,4,4) if len(transforms) > 0 else None 
        pbuf = np.vstack(planes).reshape(-1,4) if len(planes) > 0 else None

        log.debug("serialized CSG of height %2d into buf with %3d nodes, %3d transforms, %3d planes, meta %r " % (self.height, len(buf), len(transforms), len(planes), self.meta ))  
        assert tbuf is not None

        return buf, tbuf, pbuf



    def save(self, treedir):
        if not os.path.exists(treedir):
            os.makedirs(treedir)
        pass

        nodebuf, tranbuf, planebuf = self.serialize() 

        metapath = self.metapath(treedir)
        json.dump(self.meta,file(metapath,"w"))

        nodepath = self.nodepath(treedir)
        np.save(nodepath, nodebuf)
        pass
        if tranbuf is not None:
            tranpath = self.tranpath(treedir)
            np.save(tranpath, tranbuf)
        pass
        if planebuf is not None:
            planepath = self.planepath(treedir)
            np.save(planepath, planebuf)
        pass


    stream = property(lambda self:self.save(sys.stdout))

    @classmethod
    def tranpath(cls, treedir):
        return os.path.join(treedir,"transforms.npy") 
    @classmethod
    def planepath(cls, treedir):
        return os.path.join(treedir,"planes.npy") 
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
        assert os.path.exists(treedir), treedir
        log.info("load %s " % (treedir) )
         
        nodepath = cls.nodepath(treedir)
        metapath = cls.metapath(treedir)
        tranpath = cls.tranpath(treedir)
        planepath = cls.planepath(treedir)

        nodebuf = np.load(nodepath) 
        tranbuf = np.load(tranpath) if os.path.exists(tranpath) else None
        planebuf = np.load(planepath) if os.path.exists(planepath) else None

        totnodes = len(nodebuf)
        try:
            height = TREE_EXPECTED.index(totnodes)
        except ValueError:
            log.fatal("invalid serialization of length %d not in expected %r " % (totnodes,TREE_EXPECTED))
            assert 0

        def deserialize_r(buf, idx):
            node = cls.from_array(buf[idx]) if idx < len(buf) else None

            if node is not None and node.itransform is not None and node.itransform > 0:
                assert tranbuf is not None and node.itransform - 1 < len(tranbuf)  
                node.transform = tranbuf[node.itransform - 1]
            pass
               
            if node is not None and node.iplane is not None and node.iplane > 0:
                assert planebuf is not None and node.iplane - 1 < len(planebuf) 
                assert node.nplane > 3 and node.iplane - 1 + node.nplane <= len(planebuf)
                node.planes = planebuf[node.iplane-1:node.iplane-1+node.nplane]
            pass
 
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

    @classmethod
    def uniontree(cls, primitives, name):
        tb = TreeBuilder(primitives, operator="union")
        root = tb.root 
        root.name = name
        root.analyse()
        return root

    def __init__(self, typ_, name="", left=None, right=None, param=None, param1=None, param2=None, param3=None, boundary="", complement=False, translate=None, rotate=None, scale=None,  **kwa):
        if type(typ_) is str:
            typ = self.fromdesc(typ_)
        else:
            typ = typ_  
        pass

        assert name is not None and len(name) > 0


        type_ok = type(typ) is int and typ > -1 
        if not type_ok:
            log.fatal("entered CSG type is invalid : you probably beed to update python enums with : sysrap-;sysrap-csg-generate ")
        pass
        assert type_ok, (typ_, typ, type(typ))

        self.typ = typ
        self.name = name
        self.left = left
        self.right = right
        self.parent = None

        self.param = param
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

        self.boundary = boundary

        self.translate = translate
        self.rotate = rotate
        self.scale = scale
        self._transform = None
        self.complement = complement

        self.planes = []
        self.meta = kwa

    def _get_translate(self):
        return self._translate 
    def _set_translate(self, s):
        if s is None: s="0,0,0"
        self._translate = fromstring_(s) 
    translate = property(_get_translate, _set_translate)

    def _get_rotate(self):
        return self._rotate
    def _set_rotate(self, s):
        if s is None: s="0,0,1,0"
        self._rotate = fromstring_(s)
    rotate = property(_get_rotate, _set_rotate)

    def _get_scale(self):
        return self._scale
    def _set_scale(self, s):
        if s is None: s="1,1,1"
        self._scale = fromstring_(s)
    scale = property(_get_scale, _set_scale)

    def _get_transform(self):
        if self._transform is None: 
            self._transform = make_trs(self._translate, self._rotate, self._scale ) 
        return self._transform
    def _set_transform(self, trs):
        self._transform = trs
    transform = property(_get_transform, _set_transform)

    def _get_param(self):
        return self._param
    def _set_param(self, v):
        if self.is_primitive and v is None: v = [0,0,0,0]
        self._param = np.asarray(v, dtype=np.float32) if v is not None else None
    param = property(_get_param, _set_param)

    def _get_param1(self):
        return self._param1
    def _set_param1(self, v):
        if self.is_primitive and v is None: v = [0,0,0,0]
        self._param1 = np.asarray(v, dtype=np.float32) if v is not None else None
    param1 = property(_get_param1, _set_param1)

    def _get_param2(self):
        return self._param2
    def _set_param2(self, v):
        if self.is_primitive and v is None: v = [0,0,0,0]
        self._param2 = np.asarray(v, dtype=np.float32) if v is not None else None
    param2 = property(_get_param2, _set_param2)

    def _get_param3(self):
        return self._param3
    def _set_param3(self, v):
        if self.is_primitive and v is None: v = [0,0,0,0]
        self._param3 = np.asarray(v, dtype=np.float32) if v is not None else None
    param3 = property(_get_param3, _set_param3)



    def as_array(self, itransform=0, planeIdx=0, planeNum=0):
        """
        Both primitive and internal nodes:

        * q2.u.w : CSG type code eg CSG_UNION, CSG_DIFFERENCE, CSG_INTERSECTION, CSG_SPHERE, CSG_BOX, ... 
        * q3.u.w : 1-based transform index, 0 for None

        Primitive nodes only:

        * q0 : 4*float parameters eg center and radius for sphere

        """
        arr = np.zeros( (self.NJ, self.NK), dtype=np.float32 )
       
        if self.param is not None:  # avoid gibberish in buffer
            arr[Q0] = self.param
        pass
        if self.param1 is not None:  
            arr[Q1] = self.param1
        pass
        if self.param2 is not None:  
            arr[Q2] = self.param2
        pass
        if self.param3 is not None:  
            arr[Q3] = self.param3
        pass

        if self.transform is not None:
            assert itransform > 0, itransform  # 1-based transform index
            arr.view(np.uint32)[Q3,W] = itransform 
        pass

        if self.complement:
            # view as float in order to set signbit 0x80000000
            # do via float as this works even when the value is integer zero yielding negative zero
            # AND with 0x7fffffff to recover the transform idx
            np.copysign(arr.view(np.float32)[Q3,W:W+1], -1. , arr.view(np.float32)[Q3,W:W+1] )  
        pass

        if len(self.planes) > 0:
            assert planeIdx > 0 and planeNum > 3, (planeIdx, planeNum)  # 1-based plane index
            arr.view(np.uint32)[Q0,X] = planeIdx   # cf NNode::planeIdx
            arr.view(np.uint32)[Q0,Y] = planeNum   # cf NNode::planeNum
        pass

        arr.view(np.uint32)[Q2,W] = self.typ

        return arr

    @classmethod
    def from_array(cls, arr):
        typ = int(arr.view(np.uint32)[Q2,W])
        itransform = int(arr.view(np.uint32)[Q3,W]) if typ < cls.SPHERE else 0 

        if typ in cls.CONVEX_POLYHEDRA:
            iplane = int(arr.view(np.uint32)[Q0,X])  
            nplane = int(arr.view(np.uint32)[Q0,Y]) 
        else:
            iplane = 0 
            nplane = 0 
        pass 

        log.info("CSG.from_array typ %d %s itransform %d iplane %d nplane %d " % (typ, cls.desc(typ), itransform, iplane, nplane) )

        n = cls(typ) if typ > 0 else None
        if n is not None:
            n.itransform = itransform if itransform > 0 else None
            n.iplane = iplane if iplane > 0 else None
            n.nplane = nplane if nplane > 0 else None
        pass
        return n 

    def dump(self, msg="CSG.dump"):
        log.info(msg + " name:%s" % self.name)
        self.Dump(self)
        self.analyse()
        log.info("\n%s" % self.txt)


    @classmethod 
    def Dump(cls, node, depth=0):
        indent = "   " * depth    

        label = node.label(indent)
        content = node.content()

        sys.stderr.write( "%-50s : %s \n" % (label, content))

        if node.left and node.right:
            cls.Dump(node.left, depth+1)
            cls.Dump(node.right, depth+1)
        pass

    def label(self, indent=""):
        return "%s %s;%s " % (indent, self.desc(self.typ),self.name )

    def content(self):
        return "%r %r " % (self.param, self.param1)

    dsc = property(lambda self:"%s%s"%("!" if self.complement else "", self.desc(self.typ)[0:2]))

    def __repr__(self):
        rrep = "height:%d totnodes:%d " % (self.height, self.totnodes) if self.is_root else ""  
        dlr = ",".join(map(repr,filter(None,[self.left,self.right])))
        if len(dlr) > 0: dlr = "(%s)" % dlr 
        return "".join(filter(None, [self.dsc, dlr, rrep])) 


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

    is_convex_polyhedron = property(lambda self:self.typ in self.CONVEX_POLYHEDRA )
    is_primitive = property(lambda self:self.typ >= self.SPHERE )
    is_zero = property(lambda self:self.typ == self.ZERO )
    is_operator = property(lambda self:self.typ in [self.UNION,self.DIFFERENCE,self.INTERSECTION])
    is_lrzero = property(lambda self:self.is_operator and self.left.is_zero and self.right.is_zero )
    is_rzero = property(lambda self:self.is_operator and not(self.left.is_zero) and self.right.is_zero )
    is_lzero = property(lambda self:self.is_operator and self.left.is_zero and not(self.right.is_zero) )

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

   

def test_serialize_deserialize():
    log.info("test_serialize_deserialize")
    container = CSG("box", param=[0,0,0,1000], boundary="Rock//perfectAbsorbSurface/Vacuum" )
   
    s = CSG("sphere")
    b = CSG("box", translate="0,0,20", rotate="0,0,1,45", scale="1,2,3" )
    sub = CSG("union", left=s, right=b, boundary="Vacuum///GlassShottF2", hello="world")

    trees0 = [container, sub]

    base = "$TMP/csg_py"
    CSG.Serialize(trees0, base )
    trees1 = CSG.Deserialize(base)

    assert len(trees1) == len(trees0)

    for i in range(len(trees1)):
        assert np.all( trees0[i].transform == trees1[i].transform )

    
def test_analyse():
    log.info("test_analyse")
    s = CSG("sphere")
    b = CSG("box")
    sub = CSG("union", left=s, right=b)
    sub.analyse()


def test_treebuilder():
    log.info("test_treebuilder")
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split() + sprim.split() )
    nprim = len(primitives)

    for n in range(0,nprim):
        tb = TreeBuilder(primitives[0:n+1])
        tb.root.analyse()
        print tb.root.txt 

def test_uniontree():
    log.info("test_uniontree")
 
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split())

    for n in range(0,len(primitives)):
        root = CSG.uniontree(primitives[0:n+1], name="test_uniontree")
        print "\n",root.txt


def test_trapezoid():
    log.info("test_trapezoid")

    tr = CSG("trapezoid")
    tr.planes = CSG.CubePlanes(0.5)
    tr.boundary = "dummy"   
 
    trees0 = [tr]
    base = "$TMP/test_trapezoid"

    CSG.Serialize(trees0, base )
    trees1 = CSG.Deserialize(base)

    assert len(trees1) == len(trees0)
    tr1 = trees1[0] 

    assert np.all( tr1.planes == tr.planes )



if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    #test_serialize_deserialize()
    #test_analyse()
    #test_treebuilder()
    #test_uniontree()
    test_trapezoid()




   

    
