j1707 Startup Taking Too Long 
=================================



Issue
---------

Large 250k node j1707 geometry taking 50s to startup, as the new analytic
geometry currently done post cache.

Approach
------------

* move most of the processing pre-cache to minimize launch time
* what gets to GPU is all coming out of serialized buffers so it is emminently possible to 
  get startup time to under 10s even with enormous geometries





Where the time goes
----------------------

Expensive parts of startup::

    delta:env blyth$ op --j1707 --tracer --gltf 3 

    //  ~53s from launch to runloop 
    2017-08-28 11:52:07.690 INFO  [1181753] [OpticksQuery::dump@79] OpticksQuery::init queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2017-08-28 11:52:59.518 INFO  [1181753] [OpticksViz::renderLoop@447] enter runloop 

    // Loading gltf (creats tree internally) : 10s
    2017-08-28 11:52:08.798 INFO  [1181753] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    2017-08-28 11:52:18.578 INFO  [1181753] [NGLTF::load@62] NGLTF::load DONE

    // Importing scene tree (250k nodes) : 10s
    2017-08-28 11:52:19.304 INFO  [1181753] [NScene::init@196] NScene::init import_r START 
    2017-08-28 11:52:28.257 INFO  [1181753] [NScene::init@200] NScene::init import_r DONE 


    2017-08-28 11:52:56.773 INFO  [1181753] [SLog::operator@15] OpticksHub::OpticksHub DONE


Almost all startup time inside OpticksHub::loadGeometry::

    2017-08-28 13:31:36.826 INFO  [1211565] [OpticksHub::loadGeometry@243] OpticksHub::loadGeometry START
    2017-08-28 13:32:25.954 INFO  [1211565] [OpticksHub::loadGeometry@257] OpticksHub::loadGeometry DONE
    2017-08-28 13:32:28.545 INFO  [1211565] [OpticksViz::renderLoop@447] enter runloop 


::

    op --j1707 --tracer --gltf 3 

    2017-08-28 13:55:11.783 INFO  [1226301] [GGeo::loadGeometry@560] GGeo::loadGeometry START loaded 1
    2017-08-28 13:55:11.783 INFO  [1226301] [OpticksGeometry::loadGeometryBase@223] OpticksGeometry::loadGeometryBase START 

    2017-08-28 13:55:11.783 INFO  [1226301] [GGeo::loadFromCache@617] GGeo::loadFromCache START
    2017-08-28 13:55:12.724 INFO  [1226301] [GGeo::loadFromCache@637] GGeo::loadFromCache DONE  
     // loading of merged meshes is very quick

    2017-08-28 13:55:12.724 INFO  [1226301] [GGeo::loadFromGLTF@643] GGeo::loadFromGLTF START
    2017-08-28 13:55:12.724 INFO  [1226301] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    2017-08-28 13:55:22.413 INFO  [1226301] [NGLTF::load@62] NGLTF::load DONE
    2017-08-28 13:55:22.988 INFO  [1226301] [NScene::init@182] NScene::init START age(s) 2152114 days  24.909 num_gltf_nodes 290276
    2017-08-28 13:55:23.115 INFO  [1226301] [NScene::init@196] NScene::init import_r START 
    2017-08-28 13:55:32.104 INFO  [1226301] [NScene::init@200] NScene::init import_r DONE 
    2017-08-28 13:55:39.417 INFO  [1226301] [NScene::init@251] NScene::init DONE

    //  27 seconds of processing node tree at npy- level     

    2017-08-28 13:55:39.420 INFO  [1226301] [GScene::init@146] GScene::init START
    2017-08-28 13:55:39.420 INFO  [1226301] [GScene::importMeshes@271] GScene::importMeshes START num_meshes 35
    2017-08-28 13:55:39.471 INFO  [1226301] [GScene::importMeshes@318] GScene::importMeshes DONE num_meshes 35

    2017-08-28 13:55:39.809 INFO  [1226301] [*GScene::createVolumeTree@553] GScene::createVolumeTree START 
    2017-08-28 13:55:46.369 INFO  [1226301] [*GScene::createVolumeTree@573] GScene::createVolumeTree DONE num_nodes: 290276
    // 7 seconds

    2017-08-28 13:55:47.115 INFO  [1226301] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers.START
    2017-08-28 13:56:00.582 INFO  [1226301] [GScene::makeMergedMeshAndInstancedBuffers@973] GScene::makeMergedMeshAndInstancedBuffers DONE
    // 13 seconds 

    2017-08-28 13:56:00.957 INFO  [1226301] [GGeo::loadFromGLTF@658] GGeo::loadFromGLTF DONE
    2017-08-28 13:56:00.959 INFO  [1226301] [OpticksGeometry::loadGeometryBase@257] OpticksGeometry::loadGeometryBase DONE 
    2017-08-28 13:56:00.959 INFO  [1226301] [GGeo::loadGeometry@581] GGeo::loadGeometry DONE




All Time In Here : But what is needed subsequently ?  What about full GGeo persisting ?
----------------------------------------------------------------------------------------

::

    239 void OpticksHub::loadGeometry()
    240 {
    241     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
    242 
    243     LOG(info) << "OpticksHub::loadGeometry START" ;
    244 
    245     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
    246 
    247     m_geometry->loadGeometry();
    248 
    249     //   Lookup A and B are now set ...
    250     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    251     //      B : on GGeo loading in GGeo::setupLookup
    252 
    253     m_ggeo = m_geometry->getGGeo();
    254 
    255     m_ggeo->setComposition(m_composition);
    256 
    257     LOG(info) << "OpticksHub::loadGeometry DONE" ;
    258 }
    259 


Approach
-------------

Objective is not how to speed up this loading/parsing etc.. 
as it can be done once only and cached. 

Thus questions are:

* what does the processing yield ?
* is there persistancy handling for it ?
* where to implement caching ?

* what is actually needed to run simulation ? (ie what of GGeo is used in oxrap ?)
* what is actually needed to run vizualization ? (ie what of GGeo is used in oglrap ?)


oglrap
~~~~~~~~~~

Scene

    Looks like Scene and Renderers really only need the GMergedMesh buffers

    Scene::uploadGeometry 
    Scene::uploadGeometryGlobal
    Scene::uploadGeometryInstanced  


GUI
    materialLib, surfaceLib, flagNames ...


oxrap
~~~~~~~

OGeo
    GGeoLib access to GMergedMesh 

    OGeo::makeAnalyticGeometry(GMergedMesh* mm)
    needs GParts* 

    is that persisted ? 


GMergedMesh : looks like GParts not currently persisted 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     83 private:
     84     // transients that do not need persisting, persistables are down in GMesh
     85     unsigned     m_cur_vertices ;
     86     unsigned     m_cur_faces ;
     87     unsigned     m_cur_solid ;
     88     unsigned     m_num_csgskip ;
     89     GNode*       m_cur_base ;
     90     GParts*      m_parts ;
     91     std::map<unsigned int, unsigned int> m_mesh_usage ;
     92 



GParts : has save but no load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GParts actually is made from NCSG which is loaded, mesh combination results 
in concatenation of GParts ... so seems need GParts persisting, to 
fit in clearly with GMergedMesh persisting. 

::

     40         NCSG* tree = NCSG::FromNode( n , config  );
     41         
     42         GParts* pts = GParts::make( tree, spec, verbosity ) ;
     43         pts->dump("GPartsTest");



GScene : looks to be purely internal to GGeo analytic preparer : ie not needed beyond the cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


GGeo : is too monolithic to persist the whole shebang, but most of its libs are persisted (to some extent) already
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* aim to move GGeo users to using the libs rather than the monolith itself
* actually the move to GGeoBase is in that direction (GScene and GGeo are the two GGeoBase subclasses)



Example Log
-----------------

::


    // :set nowrap

    delta:env blyth$ op --j1707 --tracer --gltf 3 
    === op-cmdline-binary-match : finds 1st argument with associated binary : --tracer
    ubin /usr/local/opticks/lib/OTracerTest cfm --tracer cmdline --j1707 --tracer --gltf 3
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OTracerTest
    288 -rwxr-xr-x  1 blyth  staff  145944 Aug 23 12:01 /usr/local/opticks/lib/OTracerTest
    proceeding.. : /usr/local/opticks/lib/OTracerTest --j1707 --tracer --gltf 3
    dedupe skipping --tracer 
    2017-08-28 11:52:07.690 INFO  [1181753] [OpticksQuery::dump@79] OpticksQuery::init queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2017-08-28 11:52:07.692 INFO  [1181753] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest a181a603769c1f98ad927e7367c7aa51 age.tot_seconds 2064734 age.tot_minutes 34412.234 age.tot_hours 573.537 age.tot_days     23.897
    2017-08-28 11:52:07.694 WARN  [1181753] [BTree::loadTree@48] BTree.loadTree: can't find file /usr/local/opticks/opticksdata/export/juno/ChromaMaterialMap.json
    2017-08-28 11:52:07.699 FATAL [1181753] [NSensorList::read@133] NSensorList::read failed to open /usr/local/opticks/opticksdata/export/juno1707/g4_00.idmap
    2017-08-28 11:52:07.700 INFO  [1181753] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-28 11:52:08.039 INFO  [1181753] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-28 11:52:08.101 INFO  [1181753] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 index 2 version (null) existsdir 1
    2017-08-28 11:52:08.148 INFO  [1181753] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 index 3 version (null) existsdir 1
    2017-08-28 11:52:08.155 INFO  [1181753] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/4 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/4 index 4 version (null) existsdir 1
    2017-08-28 11:52:08.758 INFO  [1181753] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae
    2017-08-28 11:52:08.788 INFO  [1181753] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-28 11:52:08.788 INFO  [1181753] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 15
    2017-08-28 11:52:08.788 INFO  [1181753] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Acrylic]
    2017-08-28 11:52:08.788 INFO  [1181753] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 15,2,39,4
    2017-08-28 11:52:08.797 WARN  [1181753] [*GPmt::load@44] GPmt::load resource does not exist /usr/local/opticks/opticksdata/export/juno/GPmt/0
    2017-08-28 11:52:08.797 INFO  [1181753] [GGeo::loadAnalyticPmt@764] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path -
    2017-08-28 11:52:08.798 INFO  [1181753] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    2017-08-28 11:52:18.578 INFO  [1181753] [NGLTF::load@62] NGLTF::load DONE
    2017-08-28 11:52:19.151 INFO  [1181753] [NSceneConfig::NSceneConfig@48] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0]
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-08-28 11:52:19.152 WARN  [1181753] [NScene::load_asset_extras@301] NScene::load_asset_extras verbosity increase from scene gltf  extras_verbosity 1 m_verbosity 0
    2017-08-28 11:52:19.152 INFO  [1181753] [NScene::init@182] NScene::init START age(s) 2144731 days  24.823 num_gltf_nodes 290276
    2017-08-28 11:52:19.272 INFO  [1181753] [NScene::load_csg_metadata@336] NScene::load_csg_metadata verbosity 1 num_meshes 35
    2017-08-28 11:52:19.304 INFO  [1181753] [NScene::init@196] NScene::init import_r START 
    2017-08-28 11:52:28.257 INFO  [1181753] [NScene::init@200] NScene::init import_r DONE 
    2017-08-28 11:52:28.257 INFO  [1181753] [NScene::init@204] NScene::init triple_debug  num_gltf_nodes 290276 triple_mismatch 10932
    2017-08-28 11:52:28.431 INFO  [1181753] [NScene::postimportnd@616] NScene::postimportnd numNd 290276 num_selected 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-28 11:52:32.179 INFO  [1181753] [NScene::count_progeny_digests@990] NScene::count_progeny_digests verbosity 1 node_count 290276 digest_size 35
     (**) candidates fulfil repeat/vert cuts   
     (##) selected survive contained-repeat disqualification 
     **  ##  idx   0 pdig 68a31892bccd1741cc098d232c702605 num_pdig  36572 num_progeny      4 NScene::meshmeta mesh_id  22 lvidx  20 height  1 soname        PMT_3inch_pmt_solid0x1c9e270 lvname              PMT_3inch_log0x1c9ef80
     **      idx   1 pdig 683529bb1b0fedc340f2ebce47468395 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  26 lvidx  19 height  0 soname       PMT_3inch_cntr_solid0x1c9e640 lvname         PMT_3inch_cntr_log0x1c9f1f0
     **      idx   2 pdig c81fb13777b701cb8ce6cdb7f0661f1b num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  25 lvidx  17 height  0 soname PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvname       PMT_3inch_inner2_log0x1c9f120
     **      idx   3 pdig 83a5a282f092aa7baf6982b54227bb54 num_pdig  36572 num_progeny      0 NScene::meshmeta mesh_id  24 lvidx  16 height  0 soname PMT_3inch_inner1_solid_ell_helper0x1c9e510 lvname       PMT_3inch_inner1_log0x1c9f050
     **      idx   4 pdig 50308873a9847d1c2c2029b6c9de7eeb num_pdig  36572 num_progeny      2 NScene::meshmeta mesh_id  23 lvidx  18 height  0 soname PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0 lvname         PMT_3inch_body_log0x1c9eef0
     **      idx   5 pdig 27a989a1aeab2b96cedd2b6c4a7cba2f num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  17 lvidx  10 height  2 soname                      sMask0x1816f50 lvname                      lMask0x18170e0
     **      idx   6 pdig e39a411b54c3ce46fd382fef7f632157 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  21 lvidx  12 height  4 soname    PMT_20inch_inner2_solid0x1863010 lvname      PMT_20inch_inner2_log0x1863310
     **      idx   7 pdig 74d8ce91d143cad52fad9d3661dded18 num_pdig  17739 num_progeny      0 NScene::meshmeta mesh_id  20 lvidx  11 height  4 soname    PMT_20inch_inner1_solid0x1814a90 lvname      PMT_20inch_inner1_log0x1863280
     **      idx   8 pdig a80803364fbf92f1b083ebff420b6134 num_pdig  17739 num_progeny      2 NScene::meshmeta mesh_id  19 lvidx  13 height  3 soname      PMT_20inch_body_solid0x1813ec0 lvname        PMT_20inch_body_log0x1863160
     **      idx   9 pdig 6b1283d04ffc8a27e19f84e2bec2ddd6 num_pdig  17739 num_progeny      3 NScene::meshmeta mesh_id  18 lvidx  14 height  3 soname       PMT_20inch_pmt_solid0x1813600 lvname             PMT_20inch_log0x18631f0
     **  ##  idx  10 pdig 8cbe68d7d5c763820ff67b8088e0de98 num_pdig  17739 num_progeny      5 NScene::meshmeta mesh_id  16 lvidx  15 height  0 soname              sMask_virtual0x18163c0 lvname               lMaskVirtual0x1816910
     **  ##  idx  11 pdig ad8b68a55505a09ac7578f32418904b3 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  15 lvidx   9 height  2 soname                 sFasteners0x1506180 lvname                 lFasteners0x1506370
     **  ##  idx  12 pdig f93b8bbbac89ea22bac0bf188ba49a61 num_pdig    480 num_progeny      0 NScene::meshmeta mesh_id  14 lvidx   8 height  1 soname                     sStrut0x14ddd50 lvname                     lSteel0x14dde40
             idx  13 pdig 7e51746feafa7f2621f71943da8f603c num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  13 lvidx   6 height  1 soname                    sTarget0x14dd640 lvname                    lTarget0x14dd830
             idx  14 pdig c1cb7d90c1b21d9244fb041363a01416 num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  12 lvidx   7 height  1 soname                   sAcrylic0x14dd0a0 lvname                   lAcrylic0x14dd290
             idx  15 pdig 2a8e6c1bbc5183cd347725e7525758de num_pdig      1 num_progeny 290264 NScene::meshmeta mesh_id  11 lvidx  29 height  1 soname                sInnerWater0x14dcb00 lvname                lInnerWater0x14dccf0
             idx  16 pdig 9c629989608370c2cfcdd13000efd779 num_pdig      1 num_progeny 290265 NScene::meshmeta mesh_id  10 lvidx  30 height  1 soname             sReflectorInCD0x14dc560 lvname             lReflectorInCD0x14dc750
             idx  17 pdig d05b109737bc8db360f7c1d7c9e435ce num_pdig      1 num_progeny 290275 NScene::meshmeta mesh_id   0 lvidx  34 height  0 soname                     sWorld0x14d9850 lvname                     lWorld0x14d9c00
             idx  18 pdig 1401822f0db9e6eecdff1c2bf1ccfdc7 num_pdig      1 num_progeny 290266 NScene::meshmeta mesh_id   9 lvidx  31 height  0 soname            sOuterWaterPool0x14dbc70 lvname            lOuterWaterPool0x14dbd60
             idx  19 pdig 5b3b8c2e2e10f565302ca085917c5b6e num_pdig      1 num_progeny 290267 NScene::meshmeta mesh_id   8 lvidx  32 height  0 soname                sPoolLining0x14db2e0 lvname                lPoolLining0x14db8b0
             idx  20 pdig b0b2c346a748c9d728a3d8820ab0f4fa num_pdig      1 num_progeny 290268 NScene::meshmeta mesh_id   7 lvidx  33 height  0 soname                sBottomRock0x14dab90 lvname                   lBtmRock0x14db220
             idx  21 pdig 3d2f8900f2e49c02b481c2f717aa9020 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   6 lvidx   2 height  1 soname           Upper_Tyvek_tube0x2547990 lvname         lUpperChimneyTyvek0x2547c80
             idx  22 pdig 4e44f1ac85cd60e3caa56bfd4afb675e num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   5 lvidx   1 height  1 soname           Upper_Steel_tube0x2547890 lvname         lUpperChimneySteel0x2547bb0
             idx  23 pdig 011ecee7d295c066ae68d4396215c3d0 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id   4 lvidx   0 height  0 soname              Upper_LS_tube0x2547790 lvname            lUpperChimneyLS0x2547ae0
             idx  24 pdig 0b6f5322017121bc6a01b06429b96ce1 num_pdig      1 num_progeny      3 NScene::meshmeta mesh_id   3 lvidx   3 height  0 soname              Upper_Chimney0x25476d0 lvname              lUpperChimney0x2547a50
             idx  25 pdig 233607c26ba9bdb41341dd85c6e2d272 num_pdig      1 num_progeny      4 NScene::meshmeta mesh_id   2 lvidx   4 height  0 soname                   sExpHall0x14da850 lvname                   lExpHall0x14da8d0
             idx  26 pdig 7f1ea14cfc666324859d3ab689041406 num_pdig      1 num_progeny      5 NScene::meshmeta mesh_id   1 lvidx   5 height  0 soname                   sTopRock0x14da370 lvname                   lTopRock0x14da5a0
             idx  27 pdig 8ea531d2ec901e4d1bda3f1db96f6ff6 num_pdig      1 num_progeny      5 NScene::meshmeta mesh_id  27 lvidx  26 height  1 soname            upper_tubeTyvek0x254a890 lvname              lLowerChimney0x254aa20
             idx  28 pdig 29bdbc822df2e6c13dcf4afe6913525f num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  28 lvidx  21 height  3 soname                   unionLS10x2548db0 lvname         lLowerChimneyTyvek0x254ab60
             idx  29 pdig 70b48809e0305276c9defa82d51fb48c num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  29 lvidx  22 height  1 soname                AcrylicTube0x2548f40 lvname       lLowerChimneyAcrylic0x254ac30
             idx  30 pdig 4db87140662bd68076ef786f7163cedc num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  30 lvidx  23 height  4 soname                 unionSteel0x2549960 lvname         lLowerChimneySteel0x254ad00
             idx  31 pdig 6912d4b84d2d2e7f6cfd02bc50fe664b num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  31 lvidx  25 height  1 soname                   unionLS10x2549c00 lvname            lLowerChimneyLS0x254ad90
             idx  32 pdig 817808d063b210535f9a3ebbf173ea3d num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  32 lvidx  24 height  5 soname               unionBlocker0x254a570 lvname       lLowerChimneyBlocker0x254ae60
             idx  33 pdig e3f8899d3e08412c1a95878e3d4e9943 num_pdig      1 num_progeny      1 NScene::meshmeta mesh_id  33 lvidx  28 height  0 soname                  sSurftube0x2548170 lvname                  lSurftube0x254b8d0
             idx  34 pdig 5ff05a9d6ad1d0373d6cfaf43a9d1228 num_pdig      1 num_progeny      0 NScene::meshmeta mesh_id  34 lvidx  27 height  0 soname               svacSurftube0x254ba10 lvname               lvacSurftube0x254ba90
    2017-08-28 11:52:34.748 INFO  [1181753] [NScene::labelTree@1391] NScene::labelTree label_count (non-zero ridx labelTree_r) 290254 num_repeat_candidates 4
    2017-08-28 11:52:34.748 INFO  [1181753] [NScene::dumpRepeatCount@1429] NScene::dumpRepeatCount m_verbosity 1
     ridx   1 count 182860
     ridx   2 count 106434
     ridx   3 count   480
     ridx   4 count   480
    2017-08-28 11:52:34.748 INFO  [1181753] [NScene::dumpRepeatCount@1446] NScene::dumpRepeatCount totCount 290254
    2017-08-28 11:52:35.384 INFO  [1181753] [NScene::postimportmesh@634] NScene::postimportmesh numNd 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-28 11:52:35.384 INFO  [1181753] [BConfig::dump@39] NScene::postimportmesh.cfg eki 13
                      check_surf_containment : 0
                      check_aabb_containment : 0
                          disable_instancing : 0
                           csg_bbox_analytic : 0
                               csg_bbox_poly : 0
                            csg_bbox_parsurf : 0
                             csg_bbox_g4poly : 0
                             parsurf_epsilon : -5
                              parsurf_target : 200
                               parsurf_level : 2
                              parsurf_margin : 0
                                   verbosity : 0
                                  polygonize : 1
    2017-08-28 11:52:35.384 INFO  [1181753] [NSceneConfig::dump@72] bbox_type_string : CSG_BBOX_PARSURF
    2017-08-28 11:52:35.384 INFO  [1181753] [NScene::init@251] NScene::init DONE
    2017-08-28 11:52:35.388 INFO  [1181753] [GScene::init@146] GScene::init START
    2017-08-28 11:52:35.388 INFO  [1181753] [GScene::importMeshes@272] GScene::importMeshes START num_meshes 35
    2017-08-28 11:52:35.392 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    0 name sWorld0x14d9850
    2017-08-28 11:52:35.395 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    1 name sTopRock0x14da370
    2017-08-28 11:52:35.397 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    2 name sExpHall0x14da850
    2017-08-28 11:52:35.399 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    3 name Upper_Chimney0x25476d0
    2017-08-28 11:52:35.401 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    4 name Upper_LS_tube0x2547790
    2017-08-28 11:52:35.401 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    5 name Upper_Steel_tube0x2547890
    2017-08-28 11:52:35.401 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    6 name Upper_Tyvek_tube0x2547990
    2017-08-28 11:52:35.404 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    7 name sBottomRock0x14dab90
    2017-08-28 11:52:35.406 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    8 name sPoolLining0x14db2e0
    2017-08-28 11:52:35.409 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index    9 name sOuterWaterPool0x14dbc70
    2017-08-28 11:52:35.411 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   10 name sReflectorInCD0x14dc560
    2017-08-28 11:52:35.413 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   11 name sInnerWater0x14dcb00
    2017-08-28 11:52:35.415 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   12 name sAcrylic0x14dd0a0
    2017-08-28 11:52:35.417 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   13 name sTarget0x14dd640
    2017-08-28 11:52:35.417 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   14 name sStrut0x14ddd50
    2017-08-28 11:52:35.418 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   15 name sFasteners0x1506180
    2017-08-28 11:52:35.420 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   16 name sMask_virtual0x18163c0
    2017-08-28 11:52:35.420 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   17 name sMask0x1816f50
    2017-08-28 11:52:35.422 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   18 name PMT_20inch_pmt_solid0x1813600
    2017-08-28 11:52:35.424 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   19 name PMT_20inch_body_solid0x1813ec0
    2017-08-28 11:52:35.426 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   20 name PMT_20inch_inner1_solid0x1814a90
    2017-08-28 11:52:35.428 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   21 name PMT_20inch_inner2_solid0x1863010
    2017-08-28 11:52:35.430 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   22 name PMT_3inch_pmt_solid0x1c9e270
    2017-08-28 11:52:35.432 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   23 name PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0
    2017-08-28 11:52:35.432 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   24 name PMT_3inch_inner1_solid_ell_helper0x1c9e510
    2017-08-28 11:52:35.435 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   25 name PMT_3inch_inner2_solid_ell_helper0x1c9e5d0
    2017-08-28 11:52:35.435 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   26 name PMT_3inch_cntr_solid0x1c9e640
    2017-08-28 11:52:35.437 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   27 name upper_tubeTyvek0x254a890
    2017-08-28 11:52:35.437 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   28 name unionLS10x2548db0
    2017-08-28 11:52:35.437 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   29 name AcrylicTube0x2548f40
    2017-08-28 11:52:35.437 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   30 name unionSteel0x2549960
    2017-08-28 11:52:35.439 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   31 name unionLS10x2549c00
    2017-08-28 11:52:35.439 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   32 name unionBlocker0x254a570
    2017-08-28 11:52:35.439 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   33 name sSurftube0x2548170
    2017-08-28 11:52:35.440 INFO  [1181753] [GMeshLib::add@178] GMeshLib::add (GMesh) index   34 name svacSurftube0x254ba10
    2017-08-28 11:52:35.440 INFO  [1181753] [GScene::importMeshes@319] GScene::importMeshes DONE num_meshes 35
    2017-08-28 11:52:35.440 INFO  [1181753] [GScene::compareMeshes_GMeshBB@438] GScene::compareMeshes_GMeshBB num_meshes 35 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       85.3516                    sInnerWater0x14dcb00 lvidx  29 nsp    487                             union sphere cylinder   nds[  1]  11 . 
       85.2539                 sReflectorInCD0x14dc560 lvidx  30 nsp    490                             union sphere cylinder   nds[  1]  10 . 
       10.9004                   svacSurftube0x254ba10 lvidx  27 nsp    531                                             torus   nds[  1]  290275 . 
       10.9004                      sSurftube0x2548170 lvidx  28 nsp    296                                             torus   nds[  1]  290274 . 
       7.12817          PMT_20inch_body_solid0x1813ec0 lvidx  13 nsp    532           union difference zsphere cylinder torus   nds[17739]  977 983 989 995 1001 1007 1013 1019 1025 1031 ... 
        6.8313        PMT_20inch_inner2_solid0x1863010 lvidx  12 nsp    681         union intersection zsphere cylinder torus   nds[17739]  979 985 991 997 1003 1009 1015 1021 1027 1033 ... 
       1.85201           PMT_20inch_pmt_solid0x1813600 lvidx  14 nsp    223           union difference zsphere cylinder torus   nds[17739]  976 982 988 994 1000 1006 1012 1018 1024 1030 ... 
         1.815        PMT_20inch_inner1_solid0x1814a90 lvidx  11 nsp    391         union intersection zsphere cylinder torus   nds[17739]  978 984 990 996 1002 1008 1014 1020 1026 1032 ... 
      0.127613PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvidx  17 nsp    243                                           zsphere   nds[36572]  107411 107416 107421 107426 107431 107436 107441 107446 107451 107456 ... 
    2017-08-28 11:52:35.762 INFO  [1181753] [GScene::compareMeshes_GMeshBB@529] GScene::compareMeshes_GMeshBB num_meshes 35 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 9 frac 0.257143
    2017-08-28 11:52:35.762 INFO  [1181753] [GScene::createVolumeTree@554] GScene::createVolumeTree START  verbosity 1 query  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2017-08-28 11:52:35.763 INFO  [1181753] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-08-28 11:52:35.763 INFO  [1181753] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 15,2,39,4
    2017-08-28 11:52:35.763 WARN  [1181753] [GScene::lookupBoundarySpec@834] GScene::lookupBoundarySpec ana/tri imat/omat MISMATCH  tri  (  3, - , - ,  3)  ana  ( 10, - , - ,  3)  tri_spec Galactic///Galactic ana_spec Vacuum///Galactic spec Galactic///Galactic
    2017-08-28 11:52:42.255 INFO  [1181753] [GScene::createVolumeTree@574] GScene::createVolumeTree DONE num_nodes: 290276
    2017-08-28 11:52:42.255 INFO  [1181753] [GScene::init@165] GScene::init createVolumeTrue selected_count 290276
    2017-08-28 11:52:43.019 INFO  [1181753] [GScene::makeMergedMeshAndInstancedBuffers@920] GScene::makeMergedMeshAndInstancedBuffers.START   num_repeats 4  num_ridx 5
    2017-08-28 11:52:56.381 INFO  [1181753] [GScene::makeMergedMeshAndInstancedBuffers@974] GScene::makeMergedMeshAndInstancedBuffers DONE num_repeats 4 num_ridx (including global 0) 5 nmm_created 5 nmm 5
    2017-08-28 11:52:56.381 INFO  [1181753] [GScene::prepareVertexColors@204] GScene::prepareVertexColors START
    2017-08-28 11:52:56.381 INFO  [1181753] [GColorizer::traverse@93] GColorizer::traverse START
    2017-08-28 11:52:56.460 INFO  [1181753] [GColorizer::traverse@97] GColorizer::traverse colorized nodes 0
    2017-08-28 11:52:56.460 INFO  [1181753] [GScene::prepareVertexColors@206] GScene::prepareVertexColors DONE 
    2017-08-28 11:52:56.704 INFO  [1181753] [GTreePresent::write@108] GTreePresent::write /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GNodeLibAnalytic/GTreePresent.txt
    2017-08-28 11:52:56.707 INFO  [1181753] [GTreePresent::write@113] GTreePresent::write /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GNodeLibAnalytic/GTreePresent.txtDONE
    2017-08-28 11:52:56.707 INFO  [1181753] [Index::save@342] Index::save sname GItemIndexSource.json lname GItemIndexLocal.json itemtype GItemIndex ext .json idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/MeshIndexAnalytic
    2017-08-28 11:52:56.766 INFO  [1181753] [GScene::init@190] GScene::init DONE
    2017-08-28 11:52:56.766 INFO  [1181753] [GScene::dumpNode@95] GScene::dump_node nidx   3158 FOUND 

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3158:  2: 16:  2: 6:   5] bnd:Water///Water
       nd.tr.t -0.756   0.585   0.292   0.000 
                0.612   0.791  -0.000   0.000 
               -0.231   0.179  -0.956   0.000 
              4507.637 -3489.170 18648.242   1.000 

      nd.gtr.t -0.756   0.585   0.292   0.000 
                0.612   0.791  -0.000   0.000 
               -0.231   0.179  -0.956   0.000 
              4507.637 -3489.170 18648.242   1.000 


    2017-08-28 11:52:56.766 INFO  [1181753] [GScene::dumpNode@95] GScene::dump_node nidx   3159 FOUND 

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  2: 17:  0: 7:   0] bnd:Water///Acrylic
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   0.000   1.000 

      nd.gtr.t -0.756   0.585   0.292   0.000 
                0.612   0.791  -0.000   0.000 
               -0.231   0.179  -0.956   0.000 
              4507.637 -3489.170 18648.242   1.000 


    2017-08-28 11:52:56.772 INFO  [1181753] [Opticks::makeSimpleTorchStep@1246] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-08-28 11:52:56.772 INFO  [1181753] [OpticksGen::targetGenstep@130] OpticksGen::targetGenstep setting frame 3153 -0.6931,0.6589,0.2923,0.0000 0.6890,0.7248,0.0000,0.0000 -0.2119,0.2014,-0.9563,0.0000 4131.5161,-3927.2988,18648.2422,1.0000
    2017-08-28 11:52:56.772 FATAL [1181753] [GenstepNPY::setPolarization@221] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,430.0000
    2017-08-28 11:52:56.773 INFO  [1181753] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2017-08-28 11:52:56.775 FATAL [1181753] [OpticksHub::configureState@200] OpticksHub::configureState NState::description /Users/blyth/.opticks/juno/State state dir /Users/blyth/.opticks/juno/State
    2017-08-28 11:52:56.777 INFO  [1181753] [OpticksViz::setupRendermode@174] OpticksViz::setupRendermode []
    2017-08-28 11:52:56.777 WARN  [1181753] [OpticksViz::setupRendermode@178] using non-standard rendermode 
    2017-08-28 11:52:56.777 INFO  [1181753] [OpticksViz::setupRendermode@189] OpticksViz::setupRendermode rmode axis,genstep,nopstep,photon,record,
    2017-08-28 11:52:56.777 WARN  [1181753] [OpticksViz::setupRestrictions@199] disable GeometryStyle  WIRE for JUNO as too slow 
    2017-08-28 11:52:56.777 INFO  [1181753] [Scene::dumpGeometryStyles@1201] Scene::setNumGeometryStyle (Scene::dumpGeometryStyles) 
    2017-08-28 11:52:57.841 INFO  [1181753] [OpticksViz::uploadGeometry@251] Opticks time 0.0000,200.0000,50.0000,0.0000 space 0.0000,0.0000,0.0000,60000.0000 wavelength 60.0000,820.0000,20.0000,760.0000
    2017-08-28 11:52:57.956 INFO  [1181753] [OpticksGeometry::setTarget@130] OpticksGeometry::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce  0 0 0 60000
    2017-08-28 11:52:57.956 INFO  [1181753] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 0.0000,0.0000,0.0000,60000.0000
    2017-08-28 11:52:57.956 INFO  [1181753] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-08-28 11:52:57.957 INFO  [1181753] [OScene::init@91] OScene::init START
    2017-08-28 11:52:58.180 INFO  [1181753] [OScene::init@108] OScene::init (OContext) stack_size_bytes: 2180
    2017-08-28 11:52:58.184 INFO  [1181753] [OFunc::convert@28] OFunc::convert ptxname solve_callable.cu.ptx ctxname solve_callable funcnames  SolveCubicCallable num_funcs 1
    2017-08-28 11:52:58.206 INFO  [1181753] [OFunc::convert@44] OFunc::convert id 1 name SolveCubicCallable
    2017-08-28 11:52:58.209 INFO  [1181753] [OGeo::convert@169] OGeo::convert START  numMergedMesh: 5
    2017-08-28 11:52:58.209 WARN  [1181753] [OGeo::makeAnalyticGeometry@473] OGeo::makeAnalyticGeometry START verbosity 1 mm 0
    2017-08-28 11:52:58.432 WARN  [1181753] [OGeo::makeAnalyticGeometry@473] OGeo::makeAnalyticGeometry START verbosity 1 mm 1
    2017-08-28 11:52:59.156 WARN  [1181753] [OGeo::makeAnalyticGeometry@473] OGeo::makeAnalyticGeometry START verbosity 1 mm 2
    2017-08-28 11:52:59.476 WARN  [1181753] [OGeo::makeAnalyticGeometry@473] OGeo::makeAnalyticGeometry START verbosity 1 mm 3
    2017-08-28 11:52:59.485 WARN  [1181753] [OGeo::makeAnalyticGeometry@473] OGeo::makeAnalyticGeometry START verbosity 1 mm 4
    2017-08-28 11:52:59.494 INFO  [1181753] [OGeo::convert@203] OGeo::convert DONE  numMergedMesh: 5
    2017-08-28 11:52:59.494 INFO  [1181753] [OGeo::dumpStats@572] OGeo::dumpStats num_stats 5
     mmIndex   0 numPrim    22 numPart   146 numTran(triples)    35 numPlan     0
     mmIndex   1 numPrim     5 numPart     7 numTran(triples)     5 numPlan     0
     mmIndex   2 numPrim     6 numPart   100 numTran(triples)    23 numPlan     0
     mmIndex   3 numPrim     1 numPart     7 numTran(triples)     2 numPlan     0
     mmIndex   4 numPrim     1 numPart     3 numTran(triples)     1 numPlan     0
    2017-08-28 11:52:59.495 INFO  [1181753] [OScene::init@166] OScene::init DONE
    2017-08-28 11:52:59.495 INFO  [1181753] [SLog::operator@15] OScene::OScene DONE
    2017-08-28 11:52:59.495 WARN  [1181753] [OpEngine::init@65] OpEngine::init skip initPropagation as tracer mode is active  
    2017-08-28 11:52:59.495 INFO  [1181753] [SLog::operator@15] OpEngine::OpEngine DONE
    2017-08-28 11:52:59.515 FATAL [1181753] [OContext::addEntry@44] OContext::addEntry P
    2017-08-28 11:52:59.515 INFO  [1181753] [SLog::operator@15] OKGLTracer::OKGLTracer DONE
    2017-08-28 11:52:59.515 INFO  [1181753] [SLog::operator@15] OKPropagator::OKPropagator DONE
    OKMgr::init
       OptiXVersion :            3080
    2017-08-28 11:52:59.515 INFO  [1181753] [SLog::operator@15] OKMgr::OKMgr DONE
    2017-08-28 11:52:59.515 INFO  [1181753] [Bookmarks::create@249] Bookmarks::create : persisting state to slot 0
    2017-08-28 11:52:59.515 INFO  [1181753] [Bookmarks::collect@273] Bookmarks::collect 0
    2017-08-28 11:52:59.518 WARN  [1181753] [OpticksViz::prepareGUI@366] App::prepareGUI NULL TimesTable 
    2017-08-28 11:52:59.518 INFO  [1181753] [OpticksViz::renderLoop@447] enter runloop 
    2017-08-28 11:52:59.708 INFO  [1181753] [OpticksViz::renderLoop@452] after frame.show() 
    2017-08-28 11:52:59.761 INFO  [1181753] [Animator::Summary@313] Composition::gui setup Animation   OFF 0/0/    0.0000
    2017-08-28 11:52:59.761 INFO  [1181753] [Animator::Summary@313] Composition::initRotator   OFF 0/0/    0.0000
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000           -0.000 
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000           -0.000 
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000           -0.000 
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000           -0.000 
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000           -0.000 
    Animator::step bump m_count 360 
    2017-08-28 11:53:40.741 INFO  [1181753] [Frame::key_pressed@703] Frame::key_pressed escape
    /Users/blyth/opticks/bin/op.sh RC 0




