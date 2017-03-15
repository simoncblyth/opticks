#!/usr/bin/env python
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 

from dd import Dddb, Parts, Union, Intersection 
from csg import CSG
from geom import Part

class Buf(np.ndarray): pass


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

        """
        data = np.zeros([len(parts),4,4],dtype=np.float32)
        for i,part in enumerate(parts):
            nodeindex = part.node.index
            index = i + 1   # 1-based index, where parent 0 means None

            #if part.parent is not None:
            #    parent = parts.index(part.parent) + 1   # lookup index of parent in parts list  
            #else:
            #    parent = 0 
            #pass

            data[i] = part.as_quads()

            if explode>0:
                dx = i*explode
                data[i][0,0] += dx
                data[i][2,0] += dx
                data[i][3,0] += dx

            data[i].view(np.int32)[1,1] = index  
            data[i].view(np.int32)[1,2] = 0      # set to boundary index in C++ ggeo-/GPmt
            data[i].view(np.int32)[1,3] = part.flags    # used in intersect_ztubs
            # use the w slot of bb min, max for typecode and solid index
            data[i].view(np.int32)[2,3] = part.typecode 
            data[i].view(np.int32)[3,3] = nodeindex   
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
    def csg_serialize(cls, csg):
        flat = []
        for cn in csg:
            flat.extend([cn])
            pr = cn.progeny()
            flat.extend(pr)
        pass
        for k,p in enumerate(flat):
            log.debug(" %s:%s " % (k, repr(p))) 

        data = np.zeros([len(flat),4,4],dtype=np.float32)
        offset = 0 

        for cn in csg:
            assert type(cn) is CSG 
            offset = CSG.serialize_r(data, offset, cn)
        pass
        log.info("csg_serialize tot flattened %s final offset %s " % (len(flat), offset))
        assert offset == len(flat)
        buf = data.view(Buf) 
        return buf
 

    @classmethod
    def save(cls, path_, buf):
        """
        ::

            delta:~ blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
            total 80
            -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
            -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt
            -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
            -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt
            -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt

            -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
            -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
            -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
            -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt
            -rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
            delta:~ blyth$ 

        """
        path = os.path.expandvars(path_)
        pdir = os.path.dirname(path)
        if not os.path.exists(pdir):
            os.makedirs(pdir)
        pass
        log.info("saving to %s shape %s " % (path_, repr(buf.shape)))
        if hasattr(buf,"boundaries"):
            names = path.replace(".npy","_boundaries.txt")
            log.info("saving boundaries to %s " % names)
            with open(names,"w") as fp:
                fp.write("\n".join(buf.boundaries)) 
            pass
        pass

        if hasattr(buf,"materials"):
            matpath = path.replace(".npy","_materials.txt")
            log.info("saving materials to %s " % matpath)
            with open(matpath,"w") as fp:
                fp.write("\n".join(buf.materials)) 
            pass
        pass

        if hasattr(buf,"lvnames"):
            lvnpath = path.replace(".npy","_lvnames.txt")
            log.info("saving lvnames to %s " % lvnpath)
            with open(lvnpath,"w") as fp:
                fp.write("\n".join(buf.lvnames)) 
            pass
        pass
        if hasattr(buf,"pvnames"):
            pvnpath = path.replace(".npy","_pvnames.txt")
            log.info("saving pvnames to %s " % pvnpath)
            with open(pvnpath,"w") as fp:
                fp.write("\n".join(buf.pvnames)) 
            pass
        pass

        if hasattr(buf,"csg"):
            csgpath = path.replace(".npy","_csg.npy")
            csgbuf = cls.csg_serialize(buf.csg)
            if csgbuf is not None:
                log.info("saving csg to %s " % csgpath)
                #log.info(csgbuf.view(np.int32))
                #log.info(csgbuf)
                np.save(csgpath, csgbuf) 
            else:
                log.warning("csgbuf is None skip saving to %s " % csgpath)
            pass
        pass
        np.save(path, buf) 


    def traverse(self):
        self.wrap.traverse()

    def __init__(self, base):
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

    tr.save("$IDPATH/GPmt/0/GPmt.npy", partsbuf)


