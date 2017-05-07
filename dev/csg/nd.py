#!/usr/bin/env python
import numpy as np
import os, logging, json, collections

log = logging.getLogger(__name__)
expand_ = lambda path:os.path.expandvars(os.path.expanduser(path))

def makedirs_(path):
    pdir = os.path.dirname(path)
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    pass
    return path 

json_load_ = lambda path:json.load(file(expand_(path)))
json_save_ = lambda path, d:json.dump(d, file(makedirs_(expand_(path)),"w"))

from opticks.ana.base import opticks_main
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.gdml import GDML
from opticks.dev.csg.translate_gdml import translate_lv


class Mh(object):
    def __init__(self, lvIdx, lvName, soName, uri=""):
        self.lvIdx = lvIdx
        self.lvName = lvName
        self.soName = soName
        self.extras = dict(lvIdx=self.lvIdx, soName=self.soName, uri=uri )

    def __repr__(self):
        return "Mh %4d : %30s %s " % (self.lvIdx, self.soName, self.lvName )

    def _get_gltf(self):
        d = {}
        d["name"] = self.lvName
        d["extras"] = self.extras
        d["primitives"] = [dict(attributes=[])]
        return d  
    gltf = property(_get_gltf)


class Nd(object):
    """
    Mimimal representation of a node tree, just holding 
    referencing indices and transforms
    """
    count = 0 
    ulv = set()
    uso = set()
    nodes = collections.OrderedDict()
    meshes = collections.OrderedDict()

    @classmethod
    def clear(cls):
        cls.count = 0 
        cls.ulv = set()
        cls.uso = set()
        cls.nodes = collections.OrderedDict()
        cls.meshes = collections.OrderedDict()

    @classmethod
    def report(cls):
        log.info(" count %d len(ulv):%d len(uso):%d " % (cls.count, len(cls.ulv), len(cls.uso)))

    @classmethod
    def build_minimal_tree(cls, target):
        def build_r(node, depth=0):
            if depth < 3:
                nd = cls.summarize(node, depth)
            else:
                nd = None 
            pass
            for child in node.children: 
                ch = build_r(child, depth+1)
                if nd is not None and ch is not None:
                    nd.children.append(ch.ndIdx)
                pass
            pass
            return nd
        pass 
        return build_r(target)


    @classmethod
    def extras_dir(cls, lvIdx):
        return os.path.join("extras", str(lvIdx) )

    @classmethod
    def summarize(cls, node, depth):
        cls.count += 1

        transform = node.pv.transform 
        nodeIdx = node.index
        lvIdx = node.lv.idx
        lvName = node.lv.name
        solidIdx = node.lv.solid.idx
        soName = node.lv.solid.name

        #assert lvIdx == solidIdx, (lvIdx, solidIdx, lvName, soName)
        cls.ulv.add(lvIdx)
        cls.uso.add(solidIdx)
        assert len(cls.ulv) == len(cls.uso)

        ndIdx = len(cls.nodes)
        if not lvIdx in cls.meshes:
            cls.meshes[lvIdx] = Mh(lvIdx, lvName, soName)
        pass
        soIdx = list(cls.meshes.iterkeys()).index(lvIdx)  # local mesh index, using lvIdx identity
        cls.meshes[lvIdx].soIdx = soIdx 
 
        name = "ndIdx:%3d,soIdx:%3d,count:%3d,depth:%d,nodeIdx:%4d,so:%s,lv:%d:%s" % (ndIdx, soIdx, cls.count, depth, nodeIdx, node.lv.solid.name, lvIdx, node.lv.name )
        #name = node.lv.name
        log.info( name ) 

        nd = cls(ndIdx, soIdx, transform, name, depth )
        assert not ndIdx in cls.nodes

        cls.nodes[ndIdx] = nd 
        return nd  

    @classmethod
    def get(cls, ndIdx):
        return cls.nodes[ndIdx]

    def _get_gltf(self):
        d = {}
        d["mesh"] = self.soIdx
        d["name"] = self.name
        if len(self.children) > 0:
            d["children"] = self.children
        pass
        d["matrix"] = self.matrix
        return d
    gltf = property(_get_gltf)


    @classmethod
    def GLTF(cls):
        gltf = {}          
        gltf["scene"] = 0 
        gltf["scenes"] = [{ "nodes":[0] }]
        gltf["asset"] = { "version":"2.0" }
        gltf["nodes"] = [node.gltf for node in cls.nodes.values()]
        gltf["meshes"] = [mesh.gltf for mesh in cls.meshes.values()]
        return gltf 

    def __init__(self, ndIdx, soIdx, transform, name, depth):
        """
        :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
        :param soIdx: local within substree so index, used for referencing to distinct solids/meshes

        """
        self.ndIdx = ndIdx
        self.soIdx = soIdx
        self.transform = transform

        self.name = name
        self.depth = depth
        self.children = []

    matrix = property(lambda self:list(map(float,self.transform.ravel())))
    brief = property(lambda self:"Nd ndIdx:%3d soIdx:%d nch:%d matrix:%s " % (self.ndIdx, self.soIdx,  len(self.children), self.matrix))

    def __repr__(self):
        indent = ".. " * self.depth 
        return "\n".join([indent + self.brief] + map(repr,map(self.get,self.children)))   


    @classmethod
    def save_extras(cls, gdir, gdml):
        gdir = expand_(gdir)
        extras_dir = os.path.join( gdir, "extras" )
        log.info("save_extras %s " % extras_dir )
        if not os.path.exists(extras_dir):
            os.makedirs(extras_dir)
        pass
        count = 0 
        for lvIdx, mh in cls.meshes.items():
            soIdx = mh.soIdx
            lvdir = os.path.join( extras_dir, "%d" % lvIdx )
            mh.extras["uri"] = os.path.relpath(lvdir, gdir)

            lv = gdml.volumes(lvIdx) 
            cn = translate_lv(lv)

            cn.save( lvdir ) 
            count += 1 
        pass
        log.info("save_extras %s  : saved %d " % (extras_dir, count) )
        

    def save(self, path, load_check=True):

        log.info("saving to %s " % path )

        gltf = Nd.GLTF()
        json_save_(path, gltf)    

        if load_check:
            gltf2 = json_load_(path)
        pass
        return gltf 





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

    nd = Nd.build_minimal_tree(target)
    Nd.report() 

    path = "$TMP/nd/scene.gltf" 

    gdir = os.path.dirname(path)
    Nd.save_extras(gdir, gdml)    # sets uri for extra external files, so must come first  

    gltf = nd.save(path)




