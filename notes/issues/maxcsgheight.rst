Max CSG Height skipping, effectively prunes progeny of nodes with overheight csg 
==================================================================================


* maxcsgheight pruning is too draconian, as dont want progeny of an overheight csg node to 
  get skipped due to the problem with its forbears

* replacing with CSG bbox is possible BUT, dont want to duplicate the bbox calc for all primitives py side, 
  so would need to apply csgheight bbox swaps within C++ NCSG/NNode tree subclass level ?

* also cannot skip structural nodes, as the level transform is required by ancestors,
  but still need way to skip the overheight geometry 

* Instead of getting into bbox placeholders, that will just get in the way... need a 
  way to make overheight CSG to become invisible ? First thought to this is is a CSG_FLAGINVISIBLE, 
  but can the prim just be skipped altogether (probably at GParts level) 
  or is the bbox of the invisible node needed ?? Dont think the bbox is needed, but the 
  structural transform definitely is needed.

* Where exactly do the structural transforms live ? 

* if not skipped, overheight CSG will yield enormous node buffers full mostly of zeros 


Maybe here is the place to do invisible skipping, as want 
to do the same thing in analytic and polygonized worlds

::

    322 void GMergedMesh::mergeSolid( GSolid* solid, bool selected, unsigned verbosity )

::

    (lldb) bt
    * thread #1: tid = 0x2c2fa8, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8d3cf9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101e9f615 libGGeo.dylib`GMergedMesh::mergeSolid(this=0x00000001784511e0, solid=0x000000014cf6f9c0, selected=true, verbosity=1) + 69 at GMergedMesh.cc:325
        frame #5: 0x0000000101ea0a30 libGGeo.dylib`GMergedMesh::traverse_r(this=0x00000001784511e0, node=0x000000014cf6f9c0, depth=0, pass=1, verbosity=1) + 1168 at GMergedMesh.cc:514
        frame #6: 0x0000000101ea0509 libGGeo.dylib`GMergedMesh::create(ridx=0, base=0x0000000000000000, root=0x000000014cf6f9c0, verbosity=1) + 1753 at GMergedMesh.cc:171
        frame #7: 0x0000000101e7b7f8 libGGeo.dylib`GGeoLib::makeMergedMesh(this=0x0000000105934710, index=0, base=0x0000000000000000, root=0x000000014cf6f9c0, verbosity=1) + 504 at GGeoLib.cc:134
        frame #8: 0x0000000101ea5f3a libGGeo.dylib`GGeo::makeMergedMesh(this=0x0000000105934480, index=0, base=0x0000000000000000, root=0x000000014cf6f9c0, verbosity=1) + 58 at GGeo.cc:513
        frame #9: 0x0000000101e995b8 libGGeo.dylib`GScene::makeMergedMeshAndInstancedBuffers(this=0x00000001463e9570) + 1416 at GScene.cc:298
        frame #10: 0x0000000101e97928 libGGeo.dylib`GScene::createInstancedMergedMeshes(this=0x00000001463e9570, delta=true) + 56 at GScene.cc:265
        frame #11: 0x0000000101e96f42 libGGeo.dylib`GScene::init(this=0x00000001463e9570) + 82 at GScene.cc:45
        frame #12: 0x0000000101e96ea1 libGGeo.dylib`GScene::GScene(this=0x00000001463e9570, ggeo=0x0000000105934480, scene=0x0000000108266ee0) + 289 at GScene.cc:33
        frame #13: 0x0000000101e96f85 libGGeo.dylib`GScene::GScene(this=0x00000001463e9570, ggeo=0x0000000105934480, scene=0x0000000108266ee0) + 37 at GScene.cc:34
        frame #14: 0x0000000101ea6fde libGGeo.dylib`GGeo::loadFromGLTF(this=0x0000000105934480) + 782 at GGeo.cc:656
        frame #15: 0x0000000101ea61be libGGeo.dylib`GGeo::loadGeometry(this=0x0000000105934480) + 430 at GGeo.cc:568
        frame #16: 0x0000000101fd6446 libOpticksGeometry.dylib`OpticksGeometry::loadGeometryBase(this=0x0000000105935930) + 1174 at OpticksGeometry.cc:242
        frame #17: 0x0000000101fd5d27 libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105935930) + 295 at OpticksGeometry.cc:191
        frame #18: 0x0000000101fd9f19 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x000000010592f250) + 409 at OpticksHub.cc:243
        frame #19: 0x0000000101fd90ad libOpticksGeometry.dylib`OpticksHub::init(this=0x000000010592f250) + 77 at OpticksHub.cc:94
        frame #20: 0x0000000101fd8fb0 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010592f250, ok=0x0000000105921a60) + 416 at OpticksHub.cc:81
        frame #21: 0x0000000101fd918d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010592f250, ok=0x0000000105921a60) + 29 at OpticksHub.cc:83
        frame #22: 0x00000001039481e6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe8f8, argc=23, argv=0x00007fff5fbfe9d8, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #23: 0x000000010394864b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe8f8, argc=23, argv=0x00007fff5fbfe9d8, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #24: 0x000000010000a93d OKTest`main(argc=23, argv=0x00007fff5fbfe9d8) + 1373 at OKTest.cc:60
        frame #25: 0x00007fff8a48b5fd libdyld.dylib`start + 1
        frame #26: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) 


Actually better place is together repsel selection, during GMergedMesh::traverse_r, which
now incorporates the analytic combination too. This approach means the transform
hierarchy stays unchanged but can skip the overheight CSG nodes. 

::

    484 void GMergedMesh::traverse_r( GNode* node, unsigned int depth, unsigned int pass, unsigned verbosity )
    485 {
    486     GSolid* solid = dynamic_cast<GSolid*>(node) ;
    487 
    488     // using repeat index labelling in the tree
    489     //  bool repsel = getIndex() == -1 || solid->getRepeatIndex() == getIndex() ;
    490 
    491     int idx = getIndex() ;
    492     assert(idx > -1 ) ;
    493 
    494     unsigned uidx = idx > -1 ? idx : UINT_MAX ;
    495     unsigned ridx = solid->getRepeatIndex() ;
    496 
    497     bool repsel =  idx == -1 || ridx == uidx ;
    498     bool selected = solid->isSelected() && repsel ;
    499 
    500     if(verbosity > 1)
    501           LOG(info)
    502                   << "GMergedMesh::traverse_r"
    503                   << " node " << node
    504                   << " solid " << solid
    505                   << " solid.pts " << solid->getParts()
    506                   << " depth " << depth
    507                   << " NumChildren " << node->getNumChildren()
    508                   << " pass " << pass
    509                   << " selected " << selected
    510                   ;
    511 
    512 
    513     switch(pass)
    514     {
    515        case PASS_COUNT:    countSolid(solid, selected, verbosity)  ;break;
    516        case PASS_MERGE:    mergeSolid(solid, selected, verbosity)  ;break;
    517                default:    assert(0)                    ;break;
    518     }
    519 
    520     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1, pass, verbosity );
    521 }




NB invisible skipping is a temporary hack until the overheight CSG 
can be balanced, as it will mess up material boundaries, so cannot
get valid physics results with invisibles : but should be able to raytrace nevetheless 



::

    delta:analytic blyth$ tgltf-;tgltf-gdml-
    args: 
    [2017-05-21 14:35:21,794] p41143 {/Users/blyth/opticks/analytic/gdml.py:959} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-05-21 14:35:21,830] p41143 {/Users/blyth/opticks/analytic/gdml.py:973} INFO - wrapping gdml element  
    [2017-05-21 14:35:22,727] p41143 {/Users/blyth/opticks/analytic/sc.py:226} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:4 nodesCount:    0 targetNode: Node  1 : dig be50 pig 9fd5 depth 1 nchild 2  
    pv:PhysVol /dd/Structure/Sites/db-rock0xc15d358
     Position mm -16520.0 -802110.0 -2110.0  Rotation deg 0.0 0.0 -122.9  
    lv:[247] Volume /dd/Geometry/Sites/lvNearSiteRock0xc030350 /dd/Materials/Rock0xc0300c8 near_rock0xc04ba08
       [705] Subtraction near_rock0xc04ba08  
         l:[703] Box near_rock_main0xc21d4f0 mm rmin 0.0 rmax 0.0  x 50000.0 y 50000.0 z 50000.0  
         r:[704] Box near_rock_void0xc21d6c8 mm rmin 0.0 rmax 0.0  x 50010.0 y 50010.0 z 12010.0  
       [35] Material /dd/Materials/Rock0xc0300c8 solid
       PhysVol /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xbf89820
     Position mm 2500.0 -500.0 7500.0  None 
       PhysVol /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xcd2fa58
     Position mm 0.0 0.0 -5150.0  None  : Position mm -16520.0 -802110.0 -2110.0   
    [2017-05-21 14:35:22,728] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:  0,soIdx:  0,lvName:/dd/Geometry/Sites/lvNearSiteRock0xc030350 
    [2017-05-21 14:35:22,728] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:  1,soIdx:  1,lvName:/dd/Geometry/Sites/lvNearHallTop0xc136890 
    [2017-05-21 14:35:22,730] p41143 {/Users/blyth/opticks/analytic/sc.py:190} WARNING - skipped node as csg.height  4 exceeds maxcsgheight 4 sc.py:add_node_gdml nodeIdx:   3 lvIdx: 0 soName:   near_top_cover_box0xc23f970 lvName:/dd/Geometry/PoolDetails/lvNearTopCover0xc137060  
    [2017-05-21 14:35:22,730] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:  2,soIdx:  2,lvName:/dd/Geometry/RPC/lvRPCMod0xbf54e60 
    [2017-05-21 14:35:22,730] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:  3,soIdx:  3,lvName:/dd/Geometry/RPC/lvRPCFoam0xc032c88 
    [2017-05-21 14:35:22,731] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:  4,soIdx:  4,lvName:/dd/Geometry/RPC/lvRPCBarCham140xbf4c6a0 
    ...
    [2017-05-21 14:35:24,129] p41143 {/Users/blyth/opticks/analytic/sc.py:190} WARNING - skipped node as csg.height  4 exceeds maxcsgheight 4 sc.py:add_node_gdml nodeIdx:3148 lvIdx:236 soName:   near_pool_dead_box0xbf8a280 lvName:/dd/Geometry/Pool/lvNearPoolDead0xc2dc490  
    [2017-05-21 14:35:24,129] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3146,soIdx: 23,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab10xcd2f750 
    [2017-05-21 14:35:24,129] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3147,soIdx: 24,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab20xcd2fa00 
    [2017-05-21 14:35:24,130] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3148,soIdx: 25,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab30xcd30548 
    [2017-05-21 14:35:24,130] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3149,soIdx: 26,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab40xcd307b8 
    [2017-05-21 14:35:24,131] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3150,soIdx: 27,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab50xcd309e0 
    [2017-05-21 14:35:24,131] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3151,soIdx: 28,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab60xcd30408 
    [2017-05-21 14:35:24,132] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3152,soIdx: 29,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab70xcd306e0 
    [2017-05-21 14:35:24,132] p41143 {/Users/blyth/opticks/analytic/sc.py:149} INFO - add_node ndIdx:3153,soIdx: 30,lvName:/dd/Geometry/RadSlabs/lvNearRadSlab80xc3a28c8 
    [2017-05-21 14:35:24,134] p41143 {/Users/blyth/opticks/analytic/sc.py:190} WARNING - skipped node as csg.height  4 exceeds maxcsgheight 4 sc.py:add_node_gdml nodeIdx:12229 lvIdx:245 soName:   near-radslab-box-90xcd31ea0 lvName:/dd/Geometry/RadSlabs/lvNearRadSlab90xc15c208  
    [2017-05-21 14:35:24,134] p41143 {/Users/blyth/opticks/analytic/sc.py:228} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:4 nodesCount: 3154  tgNd:Nd ndIdx:  0 soIdx:0 nch:2 par:-1 matrix:[-0.5431744456291199, 0.8396198749542236, 0.0, 0.0, -0.8396198749542236, -0.5431744456291199, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -16520.0, -802110.0, -2110.0, 1.0]  
    [2017-05-21 14:35:24,134] p41143 {/Users/blyth/opticks/analytic/sc.py:253} INFO - saving to /tmp/blyth/opticks/tgltf/tgltf-gdml--.gltf 




::

    163     def add_node_gdml(self, node, depth, debug=False, maxcsgheight=0):
    164 
    165         lvIdx = node.lv.idx
    166         lvName = node.lv.name
    167         soName = node.lv.solid.name
    168         transform = node.pv.transform
    169         boundary = node.boundary
    170         nodeIdx = node.index
    171 
    172         msg = "sc.py:add_node_gdml nodeIdx:%4d lvIdx:%2d soName:%30s lvName:%s " % (nodeIdx, lvIdx, soName, lvName )
    ...
    182         csg = self.translate_lv( node.lv )
    183 
    184         if maxcsgheight == 0 or csg.height < maxcsgheight:
    185             nd = self.add_node( lvIdx, lvName, soName, transform, boundary, depth )
    186             if getattr(nd.mesh,'csg',None) is None:
    187                 nd.mesh.csg = csg
    188             pass
    189         else:
    190             log.warning("skipped node as csg.height %2d exceeds maxcsgheight %d %s " % (csg.height, maxcsgheight, msg ))
    191             nd = None
    192         pass
    193         return nd


::

    206     def add_tree_gdml(self, target, maxdepth=0, maxcsgheight=4):
    207         def build_r(node, depth=0):
    208             if maxdepth == 0 or depth < maxdepth:
    209                 nd = self.add_node_gdml(node, depth, maxcsgheight=maxcsgheight)
    210                 if nd is not None:
    211                     for child in node.children:
    212                         ch = build_r(child, depth+1)
    213                         if ch is not None:
    214                             ch.parent = nd.ndIdx
    215                             nd.children.append(ch.ndIdx)
    216                         pass
    217                     pass
    218                 else:
    219                     pass
    220                 pass
    221             else:
    222                 nd = None
    223             pass
    224             return nd
    225         pass
    226         log.info("add_tree_gdml START maxdepth:%d maxcsgheight:%d nodesCount:%5d targetNode: %r " % (maxdepth, maxcsgheight, len(self.nodes), target))
    227         tg = build_r(target)
    228         log.info("add_tree_gdml DONE maxdepth:%d maxcsgheight:%d nodesCount:%5d  tgNd:%r " % (maxdepth, maxcsgheight, len(self.nodes), tg))
    229         return tg



