#!/usr/bin/env python
import numpy as np
import os, logging, json

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


class Nd(object):
    """
    Mimimal representation of a node tree, just holding 
    referencing indices and transforms
    """
    count = 0 
    ulv = set()
    uso = set()
    nodes = {}
    meshes = {}

    @classmethod
    def clear(cls):
        cls.count = 0 
        cls.ulv = set()
        cls.uso = set()
        cls.nodes = {}
        cls.meshes = {}

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
    def summarize(cls, node, depth):
        cls.count += 1
        cls.ulv.add(node.lv.idx)
        cls.uso.add(node.lv.solid.idx)
        assert len(cls.ulv) == len(cls.uso)

        ndIdx = len(cls.nodes)
        soIdx = len(cls.meshes)     

        nodeIdx = node.index
        lvIdx = node.lv.idx
        transform = node.pv.transform 

        name = "count:%3d,depth:%d,nodeIdx:%4d,so:%s,lv:%d:%s" % (cls.count, depth, nodeIdx, node.lv.solid.name, lvIdx, node.lv.name )

        nd = cls(ndIdx, soIdx, transform, name, depth, nodeIdx, lvIdx)
        assert not ndIdx in cls.nodes

        cls.nodes[ndIdx] = nd 
        cls.meshes[soIdx] = { "name":node.lv.solid.name } 

        return nd  

    @classmethod
    def get(cls, ndIdx):
        return cls.registry[ndIdx]

    @classmethod
    def GLTF(cls):
        gltf = {}          
        gltf["asset"] = { "version":"2.0" }
        gltf["scenes"] = [{ "nodes":[0] }]
        gltf["nodes"] = [node.gltf for node in cls.nodes.values()]
        gltf["meshes"] = [mesh for mesh in cls.meshes]

        return gltf 

    def __init__(self, ndIdx, soIdx, transform, name, depth, nodeIdx, lvIdx):
        """
        :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
        :param soIdx: local within substree ms index, used for referencing to distinct solids/meshes

        :param nodeIdx: absolute (entire GDML tree) treebase.node index
        :param lvIdx:
        """
        self.ndIdx = ndIdx
        self.soIdx = soIdx

        self.nodeIdx= nodeIdx
        self.lvIdx = lvIdx 
        self.transform = transform
        self.name = name
        self.depth = depth
        self.children = []

    def _get_rprogeny(self):
        idx = []
        def progeny_r(ndIdx):
            idx.append(ndIdx)
            nd = self.get(ndIdx)
            for chIdx in nd.children:
                progeny_r(chIdx)
            pass
        pass
        progeny_r(self.ndIdx)
        return idx
    rprogeny = property(_get_rprogeny)  


    def _get_gltf(self):
        d = {}
        if self.soIdx > -1:
            d["mesh"] = self.soIdx
        pass
        d["name"] = self.name
        d["children"] = self.children
        d["matrix"] = self.matrix
        return d
    gltf = property(_get_gltf)

    nd_children = property(lambda self:map(lambda ch:self.get(ch), self.children))
    ch_unique_lv = property(lambda self:list(set(map(lambda ch:ch.lvIdx, self.nd_children))))
    smatrix = property(lambda self:",".join(map(lambda row:",".join(map(lambda v:"%10s" % v,row)), self.transform ))) 
    matrix = property(lambda self:list(map(float,self.transform.ravel())))
    stransform = property(lambda self:"".join(map(lambda row:"%30s" % row, self.transform )))
    brief = property(lambda self:"Nd ndIdx:%3d nodeIdx:%5d lvIdx:%5d nch:%d chulv:%s  st:%s " % (self.ndIdx, self.nodeIdx, self.lvIdx,  len(self.children), self.ch_unique_lv, self.matrix))

    def __repr__(self):
        indent = ".. " * self.depth 
        return "\n".join([indent + self.brief] + map(repr, self.nd_children))   






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

    nd = Nd.build_minimal_tree(target)
    Nd.report() 

    gltf = Nd.GLTF()

    path = "$TMP/nd/scene.gltf" 
    log.info("saving to %s " % path )
    json_save_(path, gltf)    





