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
import os, sys, logging, json, string, numpy as np
log = logging.getLogger(__name__)

# bring in enum values from sysrap/OpticksCSG.h
from opticks.sysrap.OpticksCSG import CSG_
from opticks.analytic.glm import make_trs, to_pyline, to_codeline, to_cpplist, make_scale
from opticks.analytic.prism import make_segment, make_trapezoid, make_icosahedron
from opticks.analytic.textgrid import TextGrid
from opticks.analytic.tboolean import TBooleanBashFunction
from opticks.analytic.nnode_test_cpp import NNodeTestCPP


Q0,Q1,Q2,Q3 = 0,1,2,3
X,Y,Z,W = 0,1,2,3

TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )
TREE_PRIMITIVES = lambda height:1 << height 
TREE_EXPECTED = map(TREE_NODES, range(10))   # [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]

fromstring_  = lambda s:np.fromstring(s, dtype=np.float32, sep=",")

def from_arg_(s, dtype=np.float32,sep=","):
   r = None 
   if type(s) is str:
       r = np.fromstring(s, dtype=dtype, sep=sep)
   elif s is not None:
       r = np.asarray(s, dtype=dtype) 
   pass
   return r


class CSG(CSG_):
    """
    Serialization layout here must echo that in NCSG 
    """
    NJ, NK = 4, 4
    FILENAME = "csg.txt"
    CONVEX_POLYHEDRA = [CSG_.TRAPEZOID]

    def depth_(self, label=False):
        """Label tree nodes with their depth from the root, return maxdepth"""
        def depth_r(node, depth):
            if node is None:return depth
            #node.textra = " %s" % depth
            if label:
                node.depth = depth
            pass
            if node.left is None and node.right is None:
                return depth
            else:
                ldepth = depth_r(node.left, depth+1)
                rdepth = depth_r(node.right, depth+1)
                return max(ldepth, rdepth)
            pass
        pass
        return depth_r(self, 0) 


    def subdepth_(self, label=False):
        """label tree with *subdepth* : the max height of each node treated as a subtree"""
        def subdepth_r(node, depth):
            assert not node is None
            subdepth = node.depth_(label=False)
            if label:
                node.subdepth = subdepth 
            pass
            #node.textra = " %s" % subdepth
            pass
            if node.left is None and node.right is None:
                pass 
            else:
                subdepth_r(node.left, depth=depth+1)
                subdepth_r(node.right, depth=depth+1)
            pass
        pass
        subdepth_r(self, 0)


    def subtrees_(self, subdepth=1):
        """collect all subtrees of a particular subdepth"""
        subtrees = []
        def subtrees_r(node):
            if node is None:return
            subtrees_r(node.left)
            subtrees_r(node.right)
            # postorder visit 
            if node.subdepth == subdepth:
                subtrees.append(node)  
            pass
        pass
        subtrees_r(self)
        return subtrees


    def balance(self):
        assert hasattr(self, 'subdepth'), "balancing requires subdepth labels first"
        def balance_r(node, depth):
            leaf = node.is_leaf 
            if not leaf: 
                balance_r(node.left, depth+1)
                balance_r(node.right, depth+1)
            pass
            # postorder visit 
            if not leaf:
                print "%2d %2d %2d : %s " % (node.subdepth,node.left.subdepth,node.right.subdepth, node) 
            pass
        pass
        balance_r(self, 0)


    def positivize(self):
        """
        Using background info from Rossignac, "Blist: A Boolean list formulation of CSG trees"

        * http://www.cc.gatech.edu/~jarek/papers/Blist.pdf  
        * https://www.ics.uci.edu/~pattis/ICS-31/lectures/simplify/lecture.html

        The positive form of a CSG expression is obtained by 
        distributing any negations down the tree to end up  
        with complements in the leaves only.

        * DIFFERNCE -> "-" 
        * UNION -> "+"  "OR"
        * INTERSECTION -> "*"  "AND"
        * COMPLEMENT -> "!" 

        deMorgan Tranformations

            A - B -> A * !B
            !(A*B) -> !A + !B
            !(A+B) -> !A * !B
            !(A - B) -> !(A*!B) -> !A + B

        Example

        (A+B)(C-(D-E))                      (A+B)(C(!D+E))

                       *                                   *
                                                                        
                  +          -                        +          *
                 / \        / \                      / \        / \
                A   B      C    -                   A   B      C    +  
                               / \                                 / \
                              D   E                              !D   E

        test_positivize

            original

                         in                    
                 un              di            
             sp      sp      bo          di    
                                     bo      bo
            positivize

                         in                    
                 un              in            
             sp      sp      bo          un    
                                    !bo      bo

        """
        deMorganSwap = {CSG_.INTERSECTION:CSG_.UNION, CSG_.UNION:CSG_.INTERSECTION }

        def positivize_r(node, negate=False, depth=0):


            if node.left is None and node.right is None:
                if negate:
                    node.complement = not node.complement
                pass
            else:
                #log.info("beg: %s %s " % (node, "NEGATE" if negate else "") ) 
                if node.typ in [CSG_.INTERSECTION, CSG_.UNION]:

                    if negate:    #  !( A*B ) ->  !A + !B       !(A+B) ->     !A * !B
                        node.typ = deMorganSwap.get(node.typ, None)
                        assert node.typ
                        left_negate = True 
                        right_negate = True
                    else:        #   A * B ->  A * B         A+B ->  A+B
                        left_negate = False
                        right_negate = False
                    pass
                elif node.typ == CSG_.DIFFERENCE:

                    if negate:  #      !(A - B) -> !(A*!B) -> !A + B
                        node.typ = CSG_.UNION 
                        left_negate = True
                        right_negate = False 
                    else:
                        node.typ = CSG_.INTERSECTION    #    A - B ->  A*!B
                        left_negate = False
                        right_negate = True 
                    pass
                else:
                    assert 0, "unexpected node.typ %s " % node.typ
                pass

                #log.info("end: %s " % node ) 
                positivize_r(node.left, negate=left_negate, depth=depth+1)
                positivize_r(node.right, negate=right_negate, depth=depth+1)
            pass
        pass
        positivize_r(self)
        assert self.is_positive_form()
        self.analyse()

    def operators_(self, minsubdepth=0):
        ops = set()
        def operators_r(node, depth):
            if node.left is None and node.right is None:
                pass
            else:
                assert node.typ in [CSG_.INTERSECTION, CSG_.UNION, CSG_.DIFFERENCE]

                # preorder visit
                if node.subdepth >= minsubdepth:
                    ops.add(node.typ)
                pass

                operators_r(node.left, depth+1)
                operators_r(node.right, depth+1)
            pass
        pass
        operators_r(self, 0)
        return list(ops)


    def is_balance_disabled(self):
        disabled = [] 
        def is_balance_disabled_r(node, depth):
            if node.left is None and node.right is None:
                if node.balance_disabled:        
                    disabled.append(node)
                pass
            else:
                if node.balance_disabled:        
                    disabled.append(node)
                pass
                is_balance_disabled_r(node.left, depth+1)
                is_balance_disabled_r(node.right, depth+1)
            pass
        pass
        is_balance_disabled_r(self, 0)

        return len(disabled) > 0


    def is_positive_form(self):
        ops = self.operators_()
        return not CSG.DIFFERENCE in ops

    def is_mono_operator(self):
        ops = self.operators_()
        return len(ops) == 1 

    def is_mono_bileaf(self):
        """
        Example of a mono bileaf tree, with all opertors 
        above subdepth 1 (bileaf level) being the same, in this case: union.
        The numbers label the subdepth.::

                                                                                                       un 7            
                                                                                        un 6                     in 1    
                                                                        un 5                     in 1         cy 0     !cy 0
                                                        un 4                     in 1         cy 0     !cy 0                
                                        un 3                     in 1         cy 0     !cy 0                                
                        un 2                     in 1         cy 0     !cy 0                                                
                in 1             in 1         cy 0     !cy 0                                                                
            cy 0     !cy 0     cy 0     !cy 0              


        """ 
        subdepth = 1   # bileaf level, one step up from the primitives
        ops = self.operators_(minsubdepth=subdepth+1)  # look at operators above the bileafs 
        return len(ops) == 1 


    def primitives(self):
        prims = []
        def primitives_r(node, depth):
            if node.left is None and node.right is None:
                prims.append(node)
            else:
                primitives_r(node.left, depth+1)
                primitives_r(node.right, depth+1)
            pass
        pass
        primitives_r(self, 0)
        return prims


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


    def alabels_(self):
        """Auto label nodes with primitive code letters"""
        prims = [] 
        def alabels_r(node):
            if node is None:return 
            alabels_r(node.left)
            alabels_r(node.right)
            # postorder visit, so primitives always visited before their parents
            if node.is_primitive:
                alabel = string.ascii_lowercase[len(prims)]
                prims.append(node)
            else: 
                if node.left is None or node.right is None:
                    log.warning("alabels_ malformed binary tree") 
                    alabel = "ERR"
                else:
                    alabel = node.left.alabel + node.right.alabel
                pass
            pass
            node.alabel = alabel
            #node.textra = " %s" % alabel
        pass
        alabels_r(self) 




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
        self.height = self.depth_(label=True)
        self.parenting_()
        self.rooting_()
        self.subdepth_(label=True)
        self.alabels_()

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


    @classmethod
    def MakeConvexPolyhedron(cls, planes, verts, bbox, srcmeta, type_="convexpolyhedron"):
        """see tboolean-segment- """
        obj = CSG(type_)
        obj.planes = planes
        obj.param2[:3] = bbox[0]
        obj.param3[:3] = bbox[1]
        obj.meta.update(srcmeta)
        return obj

    @classmethod
    def MakeSegment(cls, phi0, phi1, sz, sr ):
        """see tboolean-segment- """
        planes, verts, bbox, srcmeta = make_segment(phi0,phi1,sz,sr)
        return cls.MakeConvexPolyhedron(planes, verts, bbox, srcmeta, "segment")

    @classmethod
    def MakeTrapezoid(cls, z=200, x1=160, y1=20, x2=691.02, y2=20):
        planes, verts, bbox, srcmeta = make_trapezoid(z,x1,y1,x2,y2)
        return cls.MakeConvexPolyhedron(planes, verts, bbox, srcmeta, "trapezoid")

    @classmethod
    def MakeIcosahedron(cls, scale=100.):
        planes, verts, bbox, srcmeta = make_icosahedron(scale=scale)
        return cls.MakeConvexPolyhedron(planes, verts, bbox, srcmeta, "trapezoid")


    @classmethod
    def MakeEllipsoid(cls, axes=[1,1,1], name="MakeEllipsoid", zcut1=None, zcut2=None):

        axyz = np.asarray(axes, dtype=np.float32)

        ax = float(axyz[0])
        by = float(axyz[1])
        cz = float(axyz[2])

        zcut2 =  cz if zcut2 is None else float(zcut2)
        zcut1 = -cz if zcut1 is None else float(zcut1)


        srcmeta = dict(src_type="ellipsoid", src_ax=ax, src_by=by, src_cz=cz, src_zcut1=zcut1, src_zcut2=zcut2 )

        scale = axyz/cz    ## NB scale is 1 in z, for simple z1/z2
        z2 = zcut2
        z1 = zcut1


        cn = CSG("zsphere", name=name)

        cn.param[0] = 0
        cn.param[1] = 0
        cn.param[2] = 0
        cn.param[3] = cz

        cn.param1[0] = z1
        cn.param1[1] = z2
        cn.param1[2] = 0
        cn.param1[3] = 0

        cn.scale = scale

        cn.meta.update(srcmeta)
 
        log.info("MakeEllipsoid axyz %s scale %s " % (repr(axyz), repr(cn.scale)))

        return cn



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



    def save_nodemeta(self, treedir):
        def save_nodemeta_r(node, idx):
            nodemetapath = self.metapath(treedir,idx)
            nodemeta = {}
            nodemeta.update(node.meta)
            nodemeta.update(idx=idx)

            dir_ = os.path.dirname(nodemetapath)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            pass 
            json.dump(nodemeta,file(nodemetapath,"w"))

            if node.left is not None and node.right is not None:
                save_nodemeta_r(node.left, 2*idx+1)
                save_nodemeta_r(node.right, 2*idx+2)
            pass
        pass
        save_nodemeta_r(self,0)


    def save(self, treedir):
        if not os.path.exists(treedir):
            os.makedirs(treedir)
        pass

        nodebuf, tranbuf, planebuf = self.serialize() 

        metapath = self.metapath(treedir)
        json.dump(self.meta,file(metapath,"w"))

        self.save_nodemeta(treedir)

        lvidx = os.path.basename(treedir)
        tboolpath = self.tboolpath(treedir, lvidx)
        self.write_tbool(lvidx, tboolpath)

        nntpath = self.nntpath(treedir, lvidx)
        self.write_NNodeTest(lvidx, nntpath)

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
    def metapath(cls, treedir, idx=-1):
        return os.path.join(treedir,"meta.json") if idx == -1 else os.path.join(treedir,str(idx),"nodemeta.json")
    @classmethod
    def nodepath(cls, treedir):
        return os.path.join(treedir,"nodes.npy") 
    @classmethod
    def tboolpath(cls, treedir, name):
        return os.path.join(treedir,"tbool%s.bash" % name) 
    @classmethod
    def nntpath(cls, treedir, name):
        return os.path.join(treedir,"NNodeTest_%s.cc" % name) 


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



    def __init__(self, typ_, name="", left=None, right=None, param=None, param1=None, param2=None, param3=None, boundary="", complement=False, translate=None, rotate=None, scale=None,  **kwa):
        if type(typ_) is str:
            typ = self.fromdesc(typ_)
        else:
            typ = typ_  
        pass

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

        if len(boundary) == 0 and getattr(self.__class__,'boundary',None) != None:
            boundary = self.__class__.boundary  
            log.debug("using defaulted CSG.boundary %s " % boundary )
        pass
        self.boundary = boundary

        self.translate = translate
        self.rotate = rotate
        self.scale = scale
        self._transform = None
        self.complement = complement
        self.balance_disabled = False
        self.planes = []


        if len(kwa) == 0 and getattr(self.__class__,'kwa',None) != None:
            kwa = self.__class__.kwa  
            log.debug("using defaulted CSG.kwa %r " % kwa  )
        pass
        self.meta = kwa



    def _get_name(self):
        """When no name is given a name based on typ is returned, but not persisted, to avoid becoming stale on changing typ"""
        noname = self._name is None or len(self._name) == 0
        return self.desc(self.typ) if noname else self._name
    def _set_name(self, name):
        self._name = name
    name = property(_get_name, _set_name)

    def _get_translate(self):
        return self._translate 
    def _set_translate(self, s):
        if s is None: s="0,0,0"
        self._translate = from_arg_(s) 
    translate = property(_get_translate, _set_translate)

    def _get_rotate(self):
        return self._rotate
    def _set_rotate(self, s):
        if s is None: s="0,0,1,0"
        self._rotate = from_arg_(s)
    rotate = property(_get_rotate, _set_rotate)

    def _get_scale(self):
        return self._scale
    def _set_scale(self, s):
        if s is None: s=[1,1,1]
        self._scale = from_arg_(s)
    scale = property(_get_scale, _set_scale)

    def _get_transform(self):
        if self._transform is None: 
            self._transform = make_trs(self._translate, self._rotate, self._scale ) 
        return self._transform
    def _set_transform(self, trs):
        self._transform = from_arg_(trs)
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
        """
        Huh: looks incomplete, not loading params
        """ 
        typ = int(arr.view(np.uint32)[Q2,W])
        itransform = int(arr.view(np.uint32)[Q3,W]) if typ < cls.SPHERE else 0 
        complement = np.signbit( arr.view(np.float32)[Q3,W:W+1] )

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
            n.complement = complement 
        pass
        return n 

    def dump(self, msg="CSG.dump", detailed=False):
        self.analyse()
        log.info(msg + " name:%s" % self.name)
        sys.stderr.write("%s" % repr(self) + "\n")
        if detailed:
            self.Dump(self)
        pass
        sys.stderr.write("\n%s\n" % self.txt)





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

    def content_operator(self, lang="py"):
        fmt = "left=%s, right=%s" if lang == "py" else "&%s, &%s" 
        return fmt % ( self.left.alabel, self.right.alabel )  

    def content_operator_parenting(self, lang="py"):
        if lang == "py":
            return None
        pass
        return "%s.parent = &%s ; %s.parent = &%s ; " % (self.left.alabel, self.alabel, self.right.alabel, self.alabel )


    @classmethod
    def num_param_quads(cls, typ):
        npq = None
        if typ in [cls.CONVEXPOLYHEDRON, cls.TRAPEZOID, cls.SEGMENT]:
            npq = 0
        elif typ in [cls.SPHERE, cls.CONE, cls.BOX3, cls.BOX, cls.PLANE, cls.TORUS]:
            npq = 1
        elif typ in [cls.ZSPHERE, cls.CYLINDER, cls.DISC, cls.SLAB]:
            npq = 2 
        else:
            assert 0, "add num_param_quads for typ %s " % cls.desc(typ) 
        pass
        return npq 

    def content_param(self, lang="py"):
        if lang == "cpp" and self.num_param_quads(self.typ) == 0: 
            return None 

        if self.param is not None:
            param = to_codeline(self.param, "param" if lang == "py" else None, lang=lang) 
        else:
            param = None
        pass
        return param 

    def content_param1(self, lang="py"):
        if lang == "cpp" and self.num_param_quads(self.typ) < 2:   
            return None 

        if self.param1 is not None:
            param1 = to_codeline(self.param1, "param1" if lang == "py" else None, lang=lang) 
        else:
            param1 = None
        pass
        return param1

    def content_complement(self, lang="py"):
        if self.complement:
            if lang == "py":
                complement = "complement = True"
            else:
                complement = None 
            pass
        else:
            complement = None 
        pass
        return complement

    def content_complement_extras(self, lang="py"):
        if self.complement:
            if lang == "py":
                complement = None
            else:
                complement = "%s.complement = true" % self.alabel
            pass
        else:
            complement = None 
        pass
        return complement


    def content_transform(self, lang="py"):
        if lang == "py":
            line = to_codeline(self.transform, "%s.transform" % self.alabel, lang=lang )  
        else:
            line = to_codeline(self.transform, "%s.transform" % self.alabel, lang=lang )
        pass
        return line

    def content_type(self, lang="py"):
        if lang == "py":
            return ""
        pass
        if self.typ in [self.BOX3]:
            type_ = "nbox" 
        elif self.typ in [self.TRAPEZOID,self.SEGMENT,self.CONVEXPOLYHEDRON]:
            type_ = "nconvexpolyhedron" 
        else:
            type_ = "n%s" % self.desc(self.typ)
        pass
        return type_

    def content_maketype(self, lang="py"):
        if lang == "py":
            return self.desc(self.typ)
        pass
        if self.typ in [self.TRAPEZOID,self.SEGMENT,self.CONVEXPOLYHEDRON]:
            maketype_ = "convexpolyhedron" 
        else:
            maketype_ = "%s" % self.desc(self.typ)
        pass
        return maketype_


    def content_primitive(self, lang="py"):
        param = self.content_param(lang)
        param1 = self.content_param1(lang)
        complement = self.content_complement(lang)
        return ",".join(filter(None,[param,param1,complement]))

    def content(self, lang="py"):
        extras = []  
        if self.is_primitive:
            content_ = self.content_primitive(lang=lang)
            complement_ = self.content_complement_extras(lang=lang)
            extras.extend(filter(None,[complement_]))
        else:
            content_ = self.content_operator(lang=lang) 
            parenting_ = self.content_operator_parenting(lang=lang) 
            extras.extend(filter(None,[parenting_]))
            assert not self.complement
        pass

        extras_ = " ; ".join(extras)

        if len(extras) > 0:
            extras_ += " ; "
        pass

        if lang == "py":
            code = "%s = CSG(\"%s\", %s)" % (self.alabel, self.desc(self.typ), content_)
        else:
            type_ = self.content_type(lang)
            maketype_ = self.content_maketype(lang)
            code = "%s %s = make_%s(%s) ; %s.label = \"%s\" ; %s  " % ( type_, self.alabel, maketype_, content_, self.alabel, self.alabel, extras_ )
        pass
        return code


    def as_python_planes(self, node): 
        assert len(node.planes) > 0
        lines = []
        lines.append("%s.planes = np.zeros( (%d,4), dtype=np.float32)" % (node.alabel, len(node.planes)))
        for i in range(len(node.planes)):
            line = to_pyline(node.planes[i], "%s.planes[%d]" % (node.alabel, i))
            lines.append(line)
        pass
        lines.append("# convexpolyhedron are defined by planes and require manual aabbox definition")
        lines.append(to_pyline(node.param2[:3], "%s.param2[:3]" % node.alabel))
        lines.append(to_pyline(node.param3[:3], "%s.param3[:3]" % node.alabel))
        lines.append("")
        return lines


    def as_cpp_planes(self, node): 
        assert len(node.planes) > 0
        lines = []
        lines.append("std::vector<glm::vec4> planes ; ")
        for i in range(len(node.planes)):
            line = "planes.push_back(glm::vec4(%s));" % to_cpplist(node.planes[i])
            lines.append(line)
        pass
        lines.append("// convexpolyhedron are defined by planes and require manual aabbox definition")
        lines.append("//  npq : %s " % self.num_param_quads(node.typ) )
        lines.append("nbbox bb = make_bbox( %s , %s ) ; " % ( to_cpplist(node.param2[:3]), to_cpplist(node.param3[:3])))
        lines.append("%s.set_planes(planes);" % node.alabel)
        lines.append("%s.set_bbox(bb);" % node.alabel )
        lines.append("")
        return lines


    def as_code(self, lang="py"):
        lines = []
        def as_code_r(node):
            if node is None:return
            as_code_r(node.left) 
            as_code_r(node.right) 

            # postorder visit
            line = node.content(lang)
            lines.append(line)    
            if node.transform is not None:
                line = node.content_transform(lang)
                lines.append(line)    
            pass

            if len(node.planes) > 0:
                plins = self.as_python_planes(node) if lang == "py" else self.as_cpp_planes(node)
                for plin in plins:
                    lines.append(plin)
                pass
            pass

            if not node.is_primitive:
                lines.append("")
            pass
        pass
        as_code_r(self) 
        return "\n".join(lines)


    def as_tbool(self, name="esr"):
        tbf = TBooleanBashFunction(name=name, root=self.alabel, body=self.as_code(lang="py")  )
        return str(tbf)

    def as_NNodeTest(self, name="esr"):
        nnt  = NNodeTestCPP(name=name, root=self.alabel, body=self.as_code(lang="cpp")  )
        return str(nnt)

    def dump_tbool(self, name):
        sys.stderr.write("\n"+self.as_tbool(name))

    def dump_NNodeTest(self, name):
        sys.stderr.write("\n"+self.as_NNodeTest(name))

    def write_tbool(self, name, path):
        file(path, "w").write("\n"+self.as_tbool(name))

    def write_NNodeTest(self, name, path):
        file(path, "w").write("\n"+self.as_NNodeTest(name))


    def _get_tag(self):
        return self.desc(self.typ)[0:2]
        #return self.name[0:2]
    tag = property(_get_tag)


    dsc = property(lambda self:"%s%s"%("!" if self.complement else "", self.tag ))

    def __repr__(self):
        rrep = " height:%d totnodes:%d " % (self.height, self.totnodes) if self.is_root else ""  
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
    is_bileaf = property(lambda self:self.left is not None and self.left.is_leaf and self.right is not None and self.right.is_leaf )
    is_leaf = property(lambda self:self.left is None and self.right is None)
    is_zero = property(lambda self:self.typ == self.ZERO )
    is_operator = property(lambda self:self.typ in [self.UNION,self.DIFFERENCE,self.INTERSECTION])
    is_lrzero = property(lambda self:self.is_operator and self.left.is_zero and self.right.is_zero )      # l-zero AND r-zero
    is_rzero = property(lambda self:self.is_operator and not(self.left.is_zero) and self.right.is_zero )  # r-zero BUT !l-zero
    is_lzero = property(lambda self:self.is_operator and self.left.is_zero and not(self.right.is_zero) )  # l-zero BUT !r-zero



    @classmethod
    def Union(cls, a, b, **kwa):
        return cls(cls.UNION, left=a, right=b, **kwa)

    @classmethod
    def Intersection(cls, a, b, **kwa):
        return cls(cls.INTERSECTION, left=a, right=b, **kwa)

    @classmethod
    def Difference(cls, a, b, **kwa):
        return cls(cls.DIFFERENCE, left=a, right=b, **kwa)


    def union(self, other):
        return CSG(self.UNION, left=self, right=other)

    def subtract(self, other):
        return CSG(self.DIFFERENCE, left=self, right=other)

    def intersect(self, other):
        return CSG(self.INTERSECTION, left=self, right=other)

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



def test_positivize():
    log.info("test_positivize")

    a = CSG("sphere", param=[0,0,-50,100] ) 
    b = CSG("sphere", param=[0,0, 50,100] ) 
    c = CSG("box", param=[0,0, 50,100] ) 
    d = CSG("box", param=[0,0, 0,100] ) 
    e = CSG("box", param=[0,0, 0,100] ) 

    ab = CSG("union", left=a, right=b )
    de = CSG("difference", left=d, right=e )
    cde = CSG("difference", left=c, right=de )

    abcde = CSG("intersection", left=ab, right=cde )

    abcde.analyse()
    print "original\n\n", abcde.txt
    print "operators: " + " ".join(map(CSG.desc, abcde.operators_()))

    abcde.positivize() 
    print "positivize\n\n", abcde.txt
    print "operators: " + " ".join(map(CSG.desc, abcde.operators_()))





def test_positivize_2():
    """
    Example from p4 of Rossignac Blist paper 

    :: 

        INFO:__main__:test_positivize_2
        original

                             di                            
             in                                      in    
          a          un              un                   g
                  b       c       d          di            
                                          e       f        
        operators: union intersection difference
        positivize

                             in                            
             in                                      un    
          a          un              in                  !g
                  b       c      !d          un            
                                         !e       f        
        operators: union intersection

    """
    log.info("test_positivize_2")

    a = CSG("sphere", param=[0,0,-50,100], name="a") 
    b = CSG("sphere", param=[0,0, 50,100], name="b") 
    c = CSG("box", param=[0,0, 50,100], name="c") 
    d = CSG("box", param=[0,0, 0,100], name="d") 
    e = CSG("box", param=[0,0, 0,100], name="e") 
    f = CSG("box", param=[0,0, 0,100], name="f") 
    g = CSG("box", param=[0,0, 0,100], name="g") 

    bc = CSG.Union(b,c)
    abc = CSG.Intersection(a, bc)
    ef = CSG.Difference(e,f)
    def_ = CSG.Union(d, ef)
    defg = CSG.Intersection(def_, g)
    abcdefg = CSG.Difference(abc, defg)

    root = abcdefg


    root.analyse()
    print "original\n\n", root.txt
    print "operators: " + " ".join(map(CSG.desc, root.operators_()))

    root.positivize() 
    print "positivize\n\n", root.txt
    print "operators: " + " ".join(map(CSG.desc, root.operators_()))






def test_subdepth():
    log.info("test_subdepth")
 
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split())

    sp,bo,co,zs,cy,tr = primitives

    root = sp - bo - co - zs - cy - tr

    root.analyse()    
    print root.txt



def test_balance():
    log.info("test_balance")
 
    sprim = "sphere box cone zsphere cylinder trapezoid"
    primitives = map(CSG, sprim.split())

    sp,bo,co,zs,cy,tr = primitives

    #root = sp - bo - co - zs - cy - tr
    #root = sp * bo * co * zs * cy * tr
    root = sp + bo + co + zs + cy + tr

    root.analyse()    
    print root.txt

    root.balance()


def test_content_generate():
    log.info("test_content_generate")

    a = CSG("sphere", param=[0,0,-50,100], name="a") 
    b = CSG("sphere", param=[0,0,-50,  99], name="b") 
    b.transform = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] 

    ab = a - b 

    ab.analyse()
    obj = ab 

    for lang in "py cpp".split():
        print lang
        #print obj.content(lang)
        print obj.as_code(lang)



def test_load(lvidx):
    idfold = os.environ['OPTICKS_IDFOLD']   # from opticks_main
    base = os.path.join(idfold, "extras") 
    tree = CSG.load(CSG.treedir(base, lvidx))      
    return tree 



if __name__ == '__main__':
    pass

    from opticks.ana.base import opticks_main
    args = opticks_main()

    


    #test_serialize_deserialize()
    #test_analyse()
    #test_trapezoid()

    #test_positivize()
    #test_positivize_2()
    #test_subdepth()

    #test_balance()
    #test_content_generate()


    RSU = 56 
    ESR = 57 
    tree = test_load(ESR)
    tree.analyse()
    print tree.txt

   

    
