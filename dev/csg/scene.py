#!/usr/bin/env python
"""

https://github.com/KhronosGroup/glTF/tree/2.0/specification/2.0


* currently there is no referencing of solids, they are just listed in place



::

    In [107]: t.filternodes_so("near_pool_ows")[0].name
    Out[107]: 'Node 3150 : dig 9ff6 pig 29c2 depth 5 nchild 2938 '

::

    In [108]: g.solids(658)
    Out[108]: 
    [658] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc356df8  
         l:[656] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc2c4a40  
         l:[654] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc21d530  
         l:[652] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12e148  
         l:[650] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xbf97a68  
         l:[648] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12de98  
         l:[646] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc357900  
         l:[644] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xc12f640  
         l:[642] Subtraction near_pool_ows-ChildFornear_pool_ows_box0xbf8c148  
         l:[640] Box near_pool_ows0xc2bc1d8 mm rmin 0.0 rmax 0.0  x 15832.0 y 9832.0 z 9912.0  
         r:[641] Box near_pool_ows_sub00xc55ebf8 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[643] Box near_pool_ows_sub10xc21e940 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[645] Box near_pool_ows_sub20xc2344b0 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[647] Box near_pool_ows_sub30xbf5f5b8 mm rmin 0.0 rmax 0.0  x 4179.41484434 y 4179.41484434 z 9922.0  
         r:[649] Box near_pool_ows_sub40xbf979e0 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[651] Box near_pool_ows_sub50xc12e0c0 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[653] Box near_pool_ows_sub60xc2a23c8 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[655] Box near_pool_ows_sub70xc21d660 mm rmin 0.0 rmax 0.0  x 4176.10113585 y 4176.10113585 z 9912.0  
         r:[657] Box near_pool_ows_sub80xc2c4b70 mm rmin 0.0 rmax 0.0  x 15824.0 y 10.0 z 9912.0  


    In [150]: s = g.solids(658)

    In [151]: s.subsolids
    Out[151]: [658, 656, 654, 652, 650, 648, 646, 644, 642, 640, 641, 643, 645, 647, 649, 651, 653, 655, 657] 

    In [153]: len(g.solids(658).subsolids)
    Out[153]: 19


    In [109]: cn = g.solids(658).as_ncsg()

    In [110]: cn
    Out[110]: di(di(di(di(di(di(di(di(di(bo ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) 

    In [111]: cn.analyse()

    In [112]: cn
    Out[112]: di(di(di(di(di(di(di(di(di(bo ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo ) ,bo )height:9 totnodes:1023  


    In [114]: print cn.txt
                                                                         di    
                                                                 di          bo
                                                         di          bo        
                                                 di          bo                
                                         di          bo                        
                                 di          bo                                
                         di          bo                                        
                 di          bo                                                
         di          bo                                                        
     bo      bo                                    


"""

import numpy as np
import os, logging, json

expand_ = lambda path:os.path.expandvars(os.path.expanduser(path))
json_load_ = lambda path:json.load(file(expand_(path)))
json_save_ = lambda path, d:json.dump(d, file(expand_(path),"w"))



log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.pmt.treebase import Tree
from opticks.ana.pmt.gdml import GDML
from opticks.ana.pmt.polyconfig import PolyConfig
from opticks.dev.csg.textgrid import TextGrid

from opticks.dev.csg.csg import CSG  


class Scene(object):
    def __init__(self, gdml, base="$TMP/dev/csg/scene", name="scene.json"):
        self.gdml = gdml
        self.base = expand_(base) 
        self.name = name

    path = property(lambda self:os.path.join(self.base, self.name))

    def _get_gltf(self):
        g = {}
        g['asset'] = dict(version="2.0", generator="scene.py", copyright="Opticks")
        return g
    gltf = property(_get_gltf)

    def save(self):
        self.save_solids()
        #self.save_materials()
        #self.save_nodes() 

    def save_solids(self):
        rdir = self.prep_reldir("solids")
        solids = self.gdml.solids 
        for solid in self.gdml.solids.values():
            cn = solid.as_ncsg()
            treedir = os.path.join(rdir, "%d" % solid.idx )
            cn.save(treedir)
        pass

    def prep_reldir(self, reldir):
        rdir = os.path.join(self.base, reldir)
        if not os.path.exists(rdir):
            os.makedirs(rdir)
        pass
        return rdir

    def save_nodes(self):
        path = self.path 
        json_save_(path, dict(self.gltf))



if __name__ == '__main__':

    args = opticks_main()

    gsel = args.gsel            # string representing target node index integer or lvname
    gmaxnode = args.gmaxnode    # limit subtree node count
    gmaxdepth = args.gmaxdepth  # limit subtree node depth from the target node
    gidx = args.gidx            # target selection index, used when the gsel-ection yields multiple nodes eg when using lvname selection 

    log.info(" gsel:%s gidx:%s gmaxnode:%s gmaxdepth:%s " % (gsel, gidx, gmaxnode, gmaxdepth))


    gdmlpath = os.environ['OPTICKS_GDMLPATH']   # set within opticks_main 
    gdml = GDML.parse(gdmlpath)


    scene = Scene(gdml)

    scene.save()


if 0:

    tree = Tree(gdml.world)

    subtree = tree.subtree(gsel, maxdepth=gmaxdepth, maxnode=gmaxnode, idx=gidx)

    log.info(" subtree %s nodes " % len(subtree) ) 




