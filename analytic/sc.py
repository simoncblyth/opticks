#!/usr/bin/env python
import numpy as np
import os, logging, collections, sys

log = logging.getLogger(__name__)


from opticks.ana.base import opticks_main, expand_, json_load_, json_save_, json_save_pretty_, splitlines_
from opticks.analytic.treebase import Tree
from opticks.analytic.treebuilder import TreeBuilder
from opticks.analytic.gdml import GDML
from opticks.analytic.csg import CSG
from opticks.analytic.glm import mdot_, mdotr_,  to_pyline
from opticks.analytic.polyconfig import PolyConfig

def log_info(msg):
    sys.stdout.write(msg)


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
    identity = np.eye(4, dtype=np.float32) 
    suppress_identity = True
    def __init__(self, ndIdx, soIdx, transform, boundary, pvname, depth, scene, selected):
        """
        :param ndIdx: local within subtree nd index, used for child/parent Nd referencing
        :param soIdx: local within substree so index, used for referencing to distinct solids/meshes

        :param boundary: string with shortnames omat/osur/isur/imat eg "OwsWater///UnstStainlessSteel"

        """
        self.ndIdx = ndIdx
        self.soIdx = soIdx
        self.transform = transform

        #self.extras = dict(boundary=boundary, pvname=pvname, selected=selected)
        #self.extras = dict(boundary=boundary, pvname=pvname)   
        # got rid of duplicated and unused, to slim the gltf
        self.extras = dict(boundary=boundary)

        self.name = pvname
        self.depth = depth
        self.scene = scene 

        self.children = []
        self.parent = -1

    def _get_matrix(self):
        m = list(map(float,self.transform.ravel()))
        if self.suppress_identity:
            is_identity = np.all( self.transform == self.identity )
            if is_identity:
                m = None 
            pass
        pass
        return m 
    matrix = property(_get_matrix)


    brief = property(lambda self:"Nd ndIdx:%3d soIdx:%d nch:%d par:%d matrix:%s " % (self.ndIdx, self.soIdx,  len(self.children), self.parent, self.matrix))


    def _get_boundary(self):
        return self.extras['boundary']
    def _set_boundary(self, bnd):
        self.extras['boundary'] = bnd
    boundary = property(_get_boundary, _set_boundary)


    def _get_parent_node(self):
        return self.scene.get_node(self.parent) if self.parent > -1 else None
    parent_node = property(_get_parent_node)
    p = property(_get_parent_node)

    def _get_hierarchy(self):
        """ starting from top, going down tree to self"""
        ancestors_ = []
        p = self.parent_node
        while p:
            ancestors_.append(p)
            p = p.parent_node
        pass
        return ancestors_[::-1] + [self]
    hierarchy = property(_get_hierarchy)

    def _get_trs(self):
        hier = self.hierarchy 
        trs_ = map(lambda n:n.transform, hier )
        return trs_
    trs = property(_get_trs)

    gtr_mdot = property(lambda self:mdot_(self.trs))
    gtr_mdotr = property(lambda self:mdotr_(self.trs))

    gtr_mdot_r = property(lambda self:mdot_(self.trs[::-1]))
    gtr_mdotr_r = property(lambda self:mdotr_(self.trs[::-1]))



    def __repr__(self):
        #indent = ".. " * self.depth 
        return "%30s %s " % (self.name, self.brief)

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
        m = self.matrix
        if m is not None:
            d["matrix"] = m
        pass
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
        self.translate_node_count = 0 
        self.add_node_count = 0 
        self.selected = []


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

    def __repr__(self):
         return "\n".join([self.brief])

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


    def add_node(self, lvIdx, lvName, pvName, soName, transform, boundary, depth, selected):

        mesh = self.add_mesh(lvIdx, lvName, soName)
        soIdx = mesh.soIdx

        ndIdx = len(self.nodes)
        name = "ndIdx:%3d,soIdx:%3d,lvName:%s" % (ndIdx, soIdx, lvName)

        #log.info("add_node %s " % name)
        assert transform is not None

        nd = Nd(ndIdx, soIdx, transform, boundary, pvName, depth, self, selected )
        nd.mesh = mesh 


        assert not ndIdx in self.nodes
        self.nodes[ndIdx] = nd 
        return nd 

    def get_node(self, ndIdx):
        return self.nodes[ndIdx]

    def get_transform(self, ndIdx):
        nd = self.get_node(ndIdx)
        return nd.gtr_mdot_r 

    def add_node_gdml(self, node, depth, debug=False):
        """
        :param node: treeified ie (PV,LV) collapsed GDML node
        :param depth: integer
        :return nd: GLTF translation of the input node
        """
        lvIdx = node.lv.idx
        lvName = node.lv.name
        pvName = node.pv.name
        soName = node.lv.solid.name
        transform = node.pv.transform 
        boundary = node.boundary
        nodeIdx = node.index
        selected = node.selected

        msg = "sc.py:add_node_gdml nodeIdx:%4d lvIdx:%2d soName:%30s lvName:%s " % (nodeIdx, lvIdx, soName, lvName )
        #print msg

        if debug:
            solidIdx = node.lv.solid.idx
            self.ulv.add(lvIdx)
            self.uso.add(solidIdx)
            assert len(self.ulv) == len(self.uso)
            sys.stderr.write(msg+"\n" + repr(transform)+"\n")
        pass

        nd = self.add_node( lvIdx, lvName, pvName, soName, transform, boundary, depth, selected )

        ## hmm: why handle csg translation at node level, its more logical to do at mesh level ?
        ##      Presumably done here as it is then easy to access the lv ?
        ##

        if getattr(nd.mesh,'csg',None) is None:
            #print msg 
            csg = self.translate_lv( node.lv, self.maxcsgheight )
            nd.mesh.csg = csg 
            self.translate_node_count += 1

            if csg.meta.get('skip',0) == 1:
                log.warning("tlv(%3d): csg.skip as height %2d > %d lvn %s lvidx %s " % (self.translate_node_count, csg.height, self.maxcsgheight, node.lv.name, node.lv.idx )) 
            pass
        pass


        if selected:
            #log.info("\n\nselected nd %s \n\n%s\n\n" % (nd,  str(nd.mesh.csg.txt) ))
            self.selected.append(nd)
        pass

        return nd


    @classmethod
    def translate_lv(cls, lv, maxcsgheight, maxcsgheight2=0 ):
        """
        NB dont be tempted to convert to node here as CSG is a mesh level thing, not node level

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

        if rawcsg is None:
            err = "translate_lv solid.as_ncsg failed for solid %r lv %r " % ( solid, lv )
            log.fatal(err)
            rawcsg = CSG.MakeUndefined(err=err,lv=lv)     
        pass 
        rawcsg.analyse()

        log.debug("translate_lv DONE %-15s height %3d csg:%s " % (solid.__class__.__name__, rawcsg.height, rawcsg.name))

        csg = cls.optimize_csg(rawcsg, maxcsgheight, maxcsgheight2 )

        polyconfig = PolyConfig(lv.shortname)
        csg.meta.update(polyconfig.meta )
        csg.meta.update(lvname=lv.name, soname=lv.solid.name, height=csg.height)  

        ### Nope pvname is not appropriate in the CSG, CSG is a mesh level tink not a node/volume level thing 

        return csg 


    @classmethod
    def optimize_csg(self, rawcsg, maxcsgheight, maxcsgheight2):
        """
        :param rawcsg:
        :param maxcsgheight:  tree balancing is for height > maxcsgheight
        :param maxcsgheight2: error is raised if balanced tree height reamains > maxcsgheight2 
        :return csg:  balanced csg tree
        """
        overheight_ = lambda csg,maxheight:csg.height > maxheight and maxheight != 0

        is_balance_disabled = rawcsg.is_balance_disabled() 

        #log.info(" %s %s " % ( is_balance_disabled, rawcsg.name ))

        is_overheight = overheight_(rawcsg, maxcsgheight)
        if is_overheight:
            if is_balance_disabled:
                log.warning("tree is_overheight but marked balance_disabled leaving raw : %s " % rawcsg.name ) 
                return rawcsg 
            else:
                log.debug("proceed to balance")
        else:
            return rawcsg 
        pass
        log.debug("optimize_csg OVERHEIGHT h:%2d maxcsgheight:%d maxcsgheight2:%d %s " % (rawcsg.height,maxcsgheight, maxcsgheight2, rawcsg.name))

        rawcsg.positivize() 

        csg = TreeBuilder.balance(rawcsg)

        log.debug("optimize_csg compressed tree from height %3d to %3d " % (rawcsg.height, csg.height ))

        #assert not overheight_(csg, maxcsgheight2)
        if overheight_(csg, maxcsgheight2):
            csg.meta.update(err="optimize_csg.overheight csg.height %s maxcsgheight:%s maxcsgheight2:%s " % (csg.height,maxcsgheight,maxcsgheight2) ) 
        pass

        return csg 



    def add_tree_gdml(self, target, maxdepth=0):
        """
        :param target: treebase.Node instance, typically the root node
 
        invoked from gdml2gltf_main, notice the two different types of node:

        node
            input treebase.Node instances, derived from the GDML parse and treeification
            to de-stripe from PV-LV-PV-LV-.. to (PV,LV)-(PV,LV)-.. 
            node.children is used to traverse the tree

        nd
            output sc.Nd instances, which correspond to the GLTF output 

        """
        self.add_node_count = 0 
        def build_r(node, depth=0):
            self.add_node_count += 1 
            if self.add_node_count % 1000 == 0:
                log.info("add_tree_gdml count %s depth %s maxdepth %s " % (self.add_node_count,depth,maxdepth ))
            pass 
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
        log.info("add_tree_gdml DONE maxdepth:%d maxcsgheight:%d nodesCount:%5d tlvCount:%d addNodeCount:%d tgNd:%r " % 
             (maxdepth, self.maxcsgheight, len(self.nodes),self.translate_node_count, self.add_node_count, tg))
        return tg

    def save_extras(self, gdir):
        gdir = expand_(gdir)
        self.dump_extras()
        extras_dir = os.path.join( gdir, "extras" )
        log.debug("save_extras %s " % extras_dir )
        if not os.path.exists(extras_dir):
            os.makedirs(extras_dir)
        pass
        btxt = []
        count = 0 
        for lvIdx, mesh in self.meshes.items():
            soIdx = mesh.soIdx
            lvdir = os.path.join( extras_dir, "%d" % lvIdx )
            uri = os.path.relpath(lvdir, gdir)
            mesh.extras["uri"] = uri
            mesh.csg.save(lvdir)
            btxt.append(uri)
            count += 1 
        pass

        log.info("save_extras %s  : saved %d " % (extras_dir, count) )

        csgtxt_path = os.path.join(extras_dir, "csg.txt")
        log.info("write %d lines to %s " % (len(btxt), csgtxt_path))
        file(csgtxt_path,"w").write("\n".join(btxt))

    def dump_extras(self):
        log.info("dump_extras %d " %  len(self.meshes)) 
        for lvIdx, mesh in self.meshes.items():
            soIdx = mesh.soIdx
            print "lv %5d so %5d " % (lvIdx, soIdx)    
        pass

    def save(self, path, load_check=True, pretty_also=False):
        gdir = os.path.dirname(path)
        log.info("saving extras in %s " % gdir )
        self.save_extras(gdir)    # sets uri for extra external files, so must come before the json gltf save

        log.info("saving gltf into %s " % path )
        gltf = self.gltf
        json_save_(path, gltf)    

        if pretty_also:
            pretty_path = path.replace(".gltf",".pretty.gltf")
            log.info("also saving to %s " % pretty_path )
            json_save_pretty_(pretty_path, gltf)    
        pass

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
            
 



def gdml2gltf_main( args ):
    """
    main used by bin/gdml2gltf.py 
    """
    # envvars are set within opticks_main
    gdmlpath = os.environ['OPTICKS_GDMLPATH']   
    gltfpath = os.environ['OPTICKS_GLTFPATH']  

    assert gltfpath.startswith("/tmp") 

    if gltfpath.startswith("/tmp"):
        pass
    else:
        assert gdmlpath.replace('.gdml','.gltf') == gltfpath 
        assert gltfpath.replace('.gltf','.gdml') == gdmlpath 
    pass

    log.info("start GDML parse")
    gdml = GDML.parse(gdmlpath)

    log.info("start treeify")
    tree = Tree(gdml.world)  

    log.info("start apply_selection")
    tree.apply_selection(args.query)   # sets node.selected "volume mask" 

    log.info("start Sc.ctor")
    sc = Sc(maxcsgheight=3)

    sc.extras["verbosity"] = 1
    sc.extras["targetnode"] = 0   # args.query.query_range[0]   # hmm get rid of this ?

    log.info("start Sc.add_tree_gdml")

    tg = sc.add_tree_gdml( tree.root, maxdepth=0)

    log.info("start Sc.add_tree_gdml DONE")

    #path = args.gltfpath
    gltf = sc.save(gltfpath)

    sc.gdml = gdml 
    sc.tree = tree

    return sc


def test_range():
    #q = "index:3159,depth:1"
    q = "range:3158:3160" 
    if q is not None: 
        os.environ['OPTICKS_QUERY']=q
    pass

    args = opticks_main()
    sc = gdml2gltf_main( args )

    nd = sc.get_node(3159)
    tx = sc.get_transform(3159)

    print nd.mesh.csg.txt
    print to_pyline(nd.gtr_mdot_r, "gtr")

    print tx


if __name__ == '__main__':
    from opticks.ana.base import opticks_main
    args = opticks_main()
    sc = gdml2gltf_main( args )



