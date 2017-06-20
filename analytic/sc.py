#!/usr/bin/env python
import numpy as np
import os, logging, collections, sys

log = logging.getLogger(__name__)


from opticks.ana.base import opticks_main, expand_, json_load_, json_save_, splitlines_
from opticks.analytic.treebase import Tree
from opticks.analytic.treebuilder import TreeBuilder
from opticks.analytic.gdml import GDML
from opticks.analytic.polyconfig import PolyConfig


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
    def __init__(self, ndIdx, soIdx, transform, boundary, name, depth, scene):
        """
        :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
        :param soIdx: local within substree so index, used for referencing to distinct solids/meshes
        """
        self.ndIdx = ndIdx
        self.soIdx = soIdx
        self.transform = transform
        self.extras = dict(boundary=boundary)

        self.name = name
        self.depth = depth
        self.scene = scene 

        self.children = []
        self.parent = -1

    matrix = property(lambda self:list(map(float,self.transform.ravel())))
    brief = property(lambda self:"Nd ndIdx:%3d soIdx:%d nch:%d par:%d matrix:%s " % (self.ndIdx, self.soIdx,  len(self.children), self.parent, self.matrix))

    def __repr__(self):
        indent = ".. " * self.depth 
        return indent + self.brief

    def __str__(self):
        indent = ".. " * self.depth 
        return "\n".join([indent + self.brief] + map(repr,map(self.scene.get_node,self.children)))   

    def find_nodes(self, lvIdx):
        nodes = []
        def find_nodes_r(node):
            if node.mesh.lvIdx == lvIdx:
                nodes.append(node)
            pass
            for child in node.children:
                find_nodes_r(self.scene.get_node(child))
            pass
        pass
        find_nodes_r(self)
        return nodes
  

    def _get_gltf(self):
        d = {}
        d["mesh"] = self.soIdx
        d["name"] = self.name
        d["extras"] = self.extras
        if len(self.children) > 0:
            d["children"] = self.children
        pass
        d["matrix"] = self.matrix
        return d
    gltf = property(_get_gltf)


class Sc(object):
    def __init__(self, maxcsgheight=4):
        self.ulv = set()
        self.uso = set()
        self.nodes = collections.OrderedDict()
        self.meshes = collections.OrderedDict()
        self.extras = {}
        self.maxcsgheight = maxcsgheight
        self.translate_lv_count = 0 

    def _get_gltf(self):
        root = 0 
        d = {}          
        d["scene"] = 0 
        d["scenes"] = [{ "nodes":[root] }]
        d["asset"] = { "version":"2.0", "extras":self.extras }
        d["nodes"] = [node.gltf for node in self.nodes.values()]
        d["meshes"] = [mesh.gltf for mesh in self.meshes.values()]
        return d
    gltf = property(_get_gltf)

    brief = property(lambda self:"Sc nodes:%d meshes:%d len(ulv):%d len(uso):%d " % (len(self.nodes), len(self.meshes), len(self.ulv), len(self.uso)))

    __repr__ = brief

    def __str__(self): 
         return "\n".join([self.brief] +  map(repr, self.meshes.items()))


    def lv2so(self, lvIdx): 
        """
        Convert from an external "mesh" index lvIdx into 
        local mesh index, using lvIdx identity
        """ 
        soIdx = list(self.meshes.iterkeys()).index(lvIdx)  
        return soIdx

    def add_mesh(self, lvIdx, lvName, soName):
        if not lvIdx in self.meshes:
            self.meshes[lvIdx] = Mh(lvIdx, lvName, soName)
            self.meshes[lvIdx].soIdx = self.lv2so(lvIdx)
        pass
        return self.meshes[lvIdx]

    def get_mesh(self, lvIdx):
        return self.meshes[lvIdx]

    def find_meshes_so(self, pfx):
        return filter(lambda mesh:mesh.soName.startswith(pfx),self.meshes.values())

    def find_meshes_lv(self, pfx):
        return filter(lambda mesh:mesh.lvName.startswith(pfx),self.meshes.values())


    def add_node(self, lvIdx, lvName, soName, transform, boundary, depth):

        mesh = self.add_mesh(lvIdx, lvName, soName)
        soIdx = mesh.soIdx

        ndIdx = len(self.nodes)
        name = "ndIdx:%3d,soIdx:%3d,lvName:%s" % (ndIdx, soIdx, lvName)

        #log.info("add_node %s " % name)
        assert transform is not None

        nd = Nd(ndIdx, soIdx, transform, boundary, name, depth, self )
        nd.mesh = mesh 


        assert not ndIdx in self.nodes
        self.nodes[ndIdx] = nd 
        return nd 

    def get_node(self, ndIdx):
        return self.nodes[ndIdx]

    def add_node_gdml(self, node, depth, debug=False):

        lvIdx = node.lv.idx
        lvName = node.lv.name
        soName = node.lv.solid.name
        transform = node.pv.transform 
        boundary = node.boundary
        nodeIdx = node.index

        msg = "sc.py:add_node_gdml nodeIdx:%4d lvIdx:%2d soName:%30s lvName:%s " % (nodeIdx, lvIdx, soName, lvName )
        #print msg

        if debug:
            solidIdx = node.lv.solid.idx
            self.ulv.add(lvIdx)
            self.uso.add(solidIdx)
            assert len(self.ulv) == len(self.uso)
            sys.stderr.write(msg+"\n" + repr(transform)+"\n")
        pass

        nd = self.add_node( lvIdx, lvName, soName, transform, boundary, depth )

        ## hmm: why handle csg translation at node level, its more logical to do at mesh level ?
        ##      Presumably done here as it is then easy to access the lv ?
        ##
        if getattr(nd.mesh,'csg',None) is None:
            #print msg 
            csg = self.translate_lv( node.lv, self.maxcsgheight )
            nd.mesh.csg = csg 
            self.translate_lv_count += 1
            if csg.meta.get('skip',0) == 1:
                log.warning("tlv(%3d): csg.skip as height %2d > %d lvn %s lvidx %s " % (self.translate_lv_count, csg.height, self.maxcsgheight, node.lv.name, node.lv.idx )) 
            pass
        pass
        return nd


    @classmethod
    def translate_lv(cls, lv, maxcsgheight, maxcsgheight2=0 ):
        """
        :param lv:
        :param maxcsgheight:  CSG trees greater than this are balanced
        :param maxcsgheight2:  required post-balanced height to avoid skipping 

        There are many `solid.as_ncsg` implementations, one for each the supported GDML solids, 
        some of them return single primitives others return boolean composites, some
        such as the Polycone invokes treebuilder to provide uniontree composites.

        """ 
        if maxcsgheight2 == 0 and maxcsgheight != 0:
            maxcsgheight2 = maxcsgheight + 1
        pass  

        solid = lv.solid
        log.debug("translate_lv START %-15s %s  " % (solid.__class__.__name__, lv.name ))

        rawcsg = solid.as_ncsg()
        rawcsg.analyse()

        log.info("translate_lv DONE %-15s height %3d csg:%s " % (solid.__class__.__name__, rawcsg.height, rawcsg.name))

        csg = cls.optimize_csg(rawcsg, maxcsgheight, maxcsgheight2 )

        polyconfig = PolyConfig(lv.shortname)
        csg.meta.update(polyconfig.meta )
        csg.meta.update(lvname=lv.shortname, soname=lv.solid.name, height=csg.height)  

        return csg 


    @classmethod
    def optimize_csg(self, rawcsg, maxcsgheight, maxcsgheight2):

        overheight_ = lambda csg,maxheight:csg.height > maxheight and maxheight != 0

        if not overheight_(rawcsg, maxcsgheight):
            return rawcsg 
        pass
        log.info("optimize_csg OVERHEIGHT h:%2d maxcsgheight:%d maxcsgheight2:%d %s " % (rawcsg.height,maxcsgheight, maxcsgheight2, rawcsg.name))

        rawcsg.positivize() 

        csg = TreeBuilder.balance(rawcsg)

        log.info("optimize_csg compressed tree from height %3d to %3d " % (rawcsg.height, csg.height ))

        assert not overheight_(csg, maxcsgheight2)

        if overheight_(csg, maxcsgheight2):
            csg.meta.update(skip=1) 
        pass

        return csg 



    def add_tree_gdml(self, target, maxdepth=0):
        def build_r(node, depth=0):
            if maxdepth == 0 or depth < maxdepth:
                nd = self.add_node_gdml(node, depth)
                assert nd is not None
                for child in node.children: 
                    ch = build_r(child, depth+1)
                    if ch is not None:
                        ch.parent = nd.ndIdx
                        nd.children.append(ch.ndIdx)
                    pass
                pass
            else:
                nd = None 
            pass
            return nd
        pass 
        log.info("add_tree_gdml START maxdepth:%d maxcsgheight:%d nodesCount:%5d" % (maxdepth, self.maxcsgheight, len(self.nodes)))
        #log.info("add_tree_gdml targetNode: %r " % (target))
        tg = build_r(target)
        log.info("add_tree_gdml DONE maxdepth:%d maxcsgheight:%d nodesCount:%5d tlvCount:%d  tgNd:%r " % (maxdepth, self.maxcsgheight, len(self.nodes),self.translate_lv_count, tg))
        return tg

    def save_extras(self, gdir):
        gdir = expand_(gdir)
        extras_dir = os.path.join( gdir, "extras" )
        log.debug("save_extras %s " % extras_dir )
        if not os.path.exists(extras_dir):
            os.makedirs(extras_dir)
        pass
        count = 0 
        for lvIdx, mesh in self.meshes.items():
            soIdx = mesh.soIdx
            lvdir = os.path.join( extras_dir, "%d" % lvIdx )
            mesh.extras["uri"] = os.path.relpath(lvdir, gdir)
            mesh.csg.save(lvdir)
            count += 1 
        pass
        log.info("save_extras %s  : saved %d " % (extras_dir, count) )
 

    def save(self, path, load_check=True):
        log.info("saving to %s " % path )
        gdir = os.path.dirname(path)
        self.save_extras(gdir)    # sets uri for extra external files, so must come before the json gltf save

        gltf = self.gltf
        json_save_(path, gltf)    
        if load_check:
            gltf2 = json_load_(path)
        pass
        return gltf



    def dump_all(self, lvns):
        log.info("dump_all lvns %d " %  len(lvns))
        for lvn in lvns:
            self.dump(lvn)
        pass

    def dump(self, lvn):
        mss = self.find_meshes_lv(lvn)
        assert len(mss) == 1
        ms = mss[0]

        tree = ms.csg 
        #tree.analyse()

        tree.dump(lvn)

        if not tree.is_positive_form():
            tree.positivize()
            tree.dump(lvn + " (converted to positive form)")
        pass

        if TreeBuilder.can_balance(tree):
            balanced = TreeBuilder.balance(tree)
            balanced.dump(lvn + " (TreeBuilder balanced form)")
        else:
            log.warning("cannot balance")
        pass
            
 




if __name__ == '__main__':

    """
    NB tgltf-gdml does not use this main ... it uses the python from tgltf-gdml--
    """

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    #gsel = "/dd/Geometry/AD/lvSST0x" 
    #gsel = "/dd/Geometry/AD/lvOIL0xbf5e0b8"
    gsel = 3153


    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))

    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 
    gdml = GDML.parse(gdmlpath)

    tree = Tree(gdml.world)
    target = tree.findnode(gsel, gidx)

    #log.info(" target node %s " % target )   

    sc = Sc(maxcsgheight=3)
    tg = sc.add_tree_gdml( target, maxdepth=0 )
   
    if args.gltfsave: 
        gltfpath = "$TMP/nd/scene.gltf"
        gltf = sc.save(gltfpath)
    else:
        pass
    pass

    if len(args.lvnlist) > 0 and os.path.exists(args.lvnlist):
        lvns = splitlines_(args.lvnlist)   
        sc.dump_all(lvns)
    pass


