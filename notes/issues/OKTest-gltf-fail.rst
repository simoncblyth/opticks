OKTest-gltf-fail missing mesh index
===================================

CONFIRMED : Suspect the problem is that gltf needs to be switched on 
when creating the geocache.

* i dont like this split between GScene and GGeo, there is one geometry the 
  analytic should stand beside the triangulated within the same GGeo instance
  and geocache 




In the process of checking that find :doc:`GMeshLib_uninitialized_transforms`


::

    epsilon:~ blyth$ lldb OKTest -- --gltf 1

    epsilon:~ blyth$ lldb OKTest -- --gltf 1
    (lldb) target create "OKTest"
    Current executable set to 'OKTest' (x86_64).
    (lldb) settings set -- target.run-args  "--gltf" "1"
    (lldb) r
    Process 10988 launched: '/usr/local/opticks/lib/OKTest' (x86_64)
    2018-06-28 09:42:23.955 INFO  [251649] [SLog::operator@20] BOpticksResource::BOpticksResource  DONE
    2018-06-28 09:42:23.956 INFO  [251649] [OpticksResource::assignDetectorName@412] OpticksResource::assignDetectorName m_detector dayabay
    2018-06-28 09:42:23.957 INFO  [251649] [OpticksHub::configure@232] OpticksHub::configure m_gltf 1
    2018-06-28 09:42:23.958 INFO  [251649] [OpticksHub::loadGeometry@383] OpticksHub::loadGeometry START
    2018-06-28 09:42:23.958 INFO  [251649] [SLog::operator@20] GGeo::GGeo  DONE
    2018-06-28 09:42:23.958 INFO  [251649] [OpticksGeometry::loadGeometry@87] OpticksGeometry::loadGeometry START 
    2018-06-28 09:42:23.958 ERROR [251649] [OpticksGeometry::loadGeometryBase@119] OpticksGeometry::loadGeometryBase START 
    2018-06-28 09:42:23.958 INFO  [251649] [GGeo::loadGeometry@531] GGeo::loadGeometry START loaded 1 gltf 1
    2018-06-28 09:42:23.960 INFO  [251649] [GMaterialLib::postLoadFromCache@72] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2018-06-28 09:42:23.960 INFO  [251649] [GMaterialLib::replaceGROUPVEL@597] GMaterialLib::replaceGROUPVEL  ni 38
    2018-06-28 09:42:23.966 INFO  [251649] [GGeoLib::loadConstituents@162] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2018-06-28 09:42:23.966 INFO  [251649] [GGeoLib::loadConstituents@169] /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
    2018-06-28 09:42:24.048 INFO  [251649] [GGeoLib::loadConstituents@218] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2018-06-28 09:42:24.134 INFO  [251649] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
    2018-06-28 09:42:24.182 INFO  [251649] [GGeo::loadAnalyticFromCache@684] GGeo::loadAnalyticFromCache START
    2018-06-28 09:42:24.373 INFO  [251649] [*OpticksResource::getSensorList@1129] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2018-06-28 09:42:24.373 INFO  [251649] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2018-06-28 09:42:24.373 INFO  [251649] [GGeoLib::loadConstituents@162] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2018-06-28 09:42:24.373 INFO  [251649] [GGeoLib::loadConstituents@169] /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
    2018-06-28 09:42:24.374 INFO  [251649] [GGeoLib::loadConstituents@218] GGeoLib::loadConstituents loaded 0 ridx ()
    2018-06-28 09:42:24.374 WARN  [251649] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/PVNames.txt
    2018-06-28 09:42:24.374 WARN  [251649] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/LVNames.txt
    2018-06-28 09:42:24.374 WARN  [251649] [*Index::load@426] Index::load FAILED to load index  idpath /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 itemtype GItemIndex Source path /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexSource.json Local path /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexLocal.json
    2018-06-28 09:42:24.374 WARN  [251649] [GItemIndex::loadIndex@176] GItemIndex::loadIndex failed for  idpath /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 reldir MeshIndexAnalytic override NULL
    2018-06-28 09:42:24.374 FATAL [251649] [GMeshLib::loadFromCache@61]  meshindex load failure 
    Assertion failed: (has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G "), function loadFromCache, file /Users/blyth/opticks/ggeo/GMeshLib.cc, line 62.
    Process 10988 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff734e6b6e <+10>: jae    0x7fff734e6b78            ; <+20>
        0x7fff734e6b70 <+12>: movq   %rax, %rdi
        0x7fff734e6b73 <+15>: jmp    0x7fff734ddb00            ; cerror_nocancel
        0x7fff734e6b78 <+20>: retq   
    Target 0: (OKTest) stopped.
    (lldb) bt
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff736b1080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff734421ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7340a1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001018e99ee libGGeo.dylib`GMeshLib::loadFromCache(this=0x000000010fa28fe0) at GMeshLib.cc:62
        frame #5: 0x00000001018e97b7 libGGeo.dylib`GMeshLib::Load(ok=0x000000010ae4a2e0, analytic=true) at GMeshLib.cc:50
        frame #6: 0x00000001018ecc67 libGGeo.dylib`GScene::GScene(this=0x000000010df87b10, ok=0x000000010ae4a2e0, ggeo=0x000000010ae53090, loaded=true) at GScene.cc:115
        frame #7: 0x00000001018ec6b4 libGGeo.dylib`GScene::GScene(this=0x000000010df87b10, ok=0x000000010ae4a2e0, ggeo=0x000000010ae53090, loaded=true) at GScene.cc:122
        frame #8: 0x00000001018ec70d libGGeo.dylib`GScene::Load(ok=0x000000010ae4a2e0, ggeo=0x000000010ae53090) at GScene.cc:74
        frame #9: 0x00000001018d686b libGGeo.dylib`GGeo::loadAnalyticFromCache(this=0x000000010ae53090) at GGeo.cc:685
        frame #10: 0x00000001018d57ca libGGeo.dylib`GGeo::loadGeometry(this=0x000000010ae53090) at GGeo.cc:552
        frame #11: 0x00000001005e9ed2 libOpticksGeo.dylib`OpticksGeometry::loadGeometryBase(this=0x000000010ae512b0) at OpticksGeometry.cc:140
        frame #12: 0x00000001005e95f4 libOpticksGeo.dylib`OpticksGeometry::loadGeometry(this=0x000000010ae512b0) at OpticksGeometry.cc:89
        frame #13: 0x00000001005ed852 libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x000000010ae4cfa0) at OpticksHub.cc:387
        frame #14: 0x00000001005ec842 libOpticksGeo.dylib`OpticksHub::init(this=0x000000010ae4cfa0) at OpticksHub.cc:175
        frame #15: 0x00000001005ec72c libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ae4cfa0, ok=0x000000010ae4a2e0) at OpticksHub.cc:157
        frame #16: 0x00000001005ec94d libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ae4cfa0, ok=0x000000010ae4a2e0) at OpticksHub.cc:156
        frame #17: 0x00000001000d2d9b libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe998, argc=3, argv=0x00007ffeefbfea70, argforced=0x0000000000000000) at OKMgr.cc:44
        frame #18: 0x00000001000d31db libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe998, argc=3, argv=0x00007ffeefbfea70, argforced=0x0000000000000000) at OKMgr.cc:52
        frame #19: 0x000000010000b995 OKTest`main(argc=3, argv=0x00007ffeefbfea70) at OKTest.cc:13
        frame #20: 0x00007fff73396015 libdyld.dylib`start + 1
    (lldb) 





::

     086 GScene::GScene( Opticks* ok, GGeo* ggeo, bool loaded )
      87     :
      88     GGeoBase(),
      89     m_ok(ok),
      90     m_query(ok->getQuery()),
      91     m_ggeo(ggeo),
      92 
      93     m_sensor_list(ok->getSensorList()),
      94     m_tri_geolib(ggeo->getGeoLib()),
      95     m_tri_mm0(m_tri_geolib->getMergedMesh(0)),
      96 
      97     m_tri_nodelib(ggeo->getNodeLib()),
      98     m_tri_bndlib(ggeo->getBndLib()),
      99     m_tri_meshlib(ggeo->getMeshLib()),
     100     m_tri_meshindex(m_tri_meshlib->getMeshIndex()),
     101 
     102 
     103     m_analytic(true),
     104     m_testgeo(false),
     105     m_loaded(loaded),
     106     m_honour_selection(true),
     107     m_gltf(m_ok->getGLTF()),
     108     m_scene_config( m_ok->getSceneConfig() ),
     109     m_scene(loaded ? NULL : (m_gltf > 0 ? NScene::Load(m_ok->getGLTFBase(), m_ok->getGLTFName(), m_ok->getIdFold(), m_scene_config, m_ok->getDbgNode()) : NULL)),
     110     m_num_nd(nd::num_nodes()),
     111     m_targetnode(m_scene ? m_scene->getTargetNode() : 0),
     112 
     113     m_geolib(loaded ? GGeoLib::Load(m_ok, m_analytic, m_tri_bndlib ) : new GGeoLib(m_ok, m_analytic, m_tri_bndlib)),
     114     m_nodelib(loaded ? GNodeLib::Load(m_ok, m_analytic, m_testgeo )  : new GNodeLib(m_ok, m_analytic, m_testgeo )),
     115     m_meshlib(loaded ? GMeshLib::Load(m_ok, m_analytic)              : new GMeshLib(m_ok, m_analytic)),
     116 
     117     m_colorizer(new GColorizer(m_nodelib, m_geolib, m_tri_bndlib, ggeo->getColors(), GColorizer::PSYCHEDELIC_NODE )),   // GColorizer::SURFACE_INDEX
     118 
     119     m_verbosity(m_scene ? m_scene->getVerbosity() : 0),
     120     m_root(NULL),
     121     m_selected_count(0)
     122 {



     071 GScene* GScene::Load(Opticks* ok, GGeo* ggeo)
      72 {
      73     bool loaded = true ;
      74     GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
      75     return scene ;
      76 }

     609 void GGeo::loadAnalyticFromGLTF()
     610 {
     611     LOG(info) << "GGeo::loadAnalyticFromGLTF START" ;
     612     if(!m_ok->isGLTF()) return ;
     613 
     614 #ifdef OPTICKS_YoctoGL
     615     m_gscene = GScene::Create(m_ok, this);
     616 #else
     617     LOG(fatal) << "GGeo::loadAnalyticFromGLTF requires YoctoGL external " ;
     618     assert(0);
     619 #endif
     620 
     621     LOG(info) << "GGeo::loadAnalyticFromGLTF DONE" ;
     622 }





     525 void GGeo::loadGeometry()
     526 {
     527     bool loaded = isLoaded() ;
     528 
     529     int gltf = m_ok->getGLTF();
     530 
     531     LOG(info) << "GGeo::loadGeometry START"
     532               << " loaded " << loaded
     533               << " gltf " << gltf
     534               ;
     535 
     536     if(!loaded)
     537     {
     538         loadFromG4DAE();
     539         save();
     540 
     541         if(gltf > 0 && gltf < 10)
     542         {
     543             loadAnalyticFromGLTF();
     544             saveAnalytic();
     545         }
     546     }
     547     else
     548     {
     549         loadFromCache();
     550         if(gltf > 0 && gltf < 10)
     551         {
     552             loadAnalyticFromCache();
     553         }
     554     }
     555 
     556 
     557     if(m_ok->isAnalyticPMTLoad())
     558     {
     559         m_pmtlib = GPmtLib::load(m_ok, m_bndlib );
     560     }
     561 
     562     if( gltf >= 10 )
     563     {
     564         LOG(info) << "GGeo::loadGeometry DEBUGGING loadAnalyticFromGLTF " ;
     565         loadAnalyticFromGLTF();
     566     }
     567 
     568     setupLookup();
     569     setupColors();
     570     setupTyp();
     571     LOG(info) << "GGeo::loadGeometry DONE" ;
     572 }

 


Compare some old geocache::

    epsilon:1 blyth$ ll /Volumes/Delta/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/
    total 0
    drwxr-xr-x    4 blyth  staff   136 Nov 28  2017 MeshIndex
    drwxr-xr-x    5 blyth  staff   170 Nov 28  2017 GSurfaceLib
    drwxr-xr-x    3 blyth  staff   102 Nov 28  2017 GSourceLib
    drwxr-xr-x    5 blyth  staff   170 Nov 28  2017 GScintillatorLib
    drwxr-xr-x    5 blyth  staff   170 Nov 28  2017 GNodeLib
    drwxr-xr-x    3 blyth  staff   102 Nov 28  2017 GMaterialLib
    drwxr-xr-x    6 blyth  staff   204 Nov 28  2017 GItemList
    drwxr-xr-x    5 blyth  staff   170 Nov 28  2017 GBndLib
    drwxr-xr-x    4 blyth  staff   136 Nov 28  2017 MeshIndexAnalytic
    drwxr-xr-x    5 blyth  staff   170 Nov 28  2017 GNodeLibAnalytic
    drwxr-xr-x   17 blyth  staff   578 Nov 28  2017 .
    drwxr-xr-x    4 blyth  staff   136 Nov 29  2017 ..
    drwxr-xr-x  251 blyth  staff  8534 Nov 29  2017 GMeshLib
    drwxr-xr-x    8 blyth  staff   272 Nov 29  2017 GMergedMesh
    drwxr-xr-x    7 blyth  staff   238 Nov 29  2017 GPartsAnalytic
    drwxr-xr-x  251 blyth  staff  8534 Nov 29  2017 GMeshLibAnalytic
    drwxr-xr-x    8 blyth  staff   272 Nov 29  2017 GMergedMeshAnalytic

    epsilon:1 blyth$ ll /Volumes/Delta/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/2/
    total 0
    drwxr-xr-x    4 blyth  staff   136 Nov 29  2017 MeshIndex
    drwxr-xr-x    5 blyth  staff   170 Nov 29  2017 GSurfaceLib
    drwxr-xr-x    3 blyth  staff   102 Nov 29  2017 GSourceLib
    drwxr-xr-x    5 blyth  staff   170 Nov 29  2017 GScintillatorLib
    drwxr-xr-x    5 blyth  staff   170 Nov 29  2017 GNodeLib
    drwxr-xr-x    3 blyth  staff   102 Nov 29  2017 GMaterialLib
    drwxr-xr-x    6 blyth  staff   204 Nov 29  2017 GItemList
    drwxr-xr-x    3 blyth  staff   102 Nov 29  2017 GBndLib
    drwxr-xr-x    4 blyth  staff   136 Nov 29  2017 ..
    drwxr-xr-x   12 blyth  staff   408 Nov 29  2017 .
    drwxr-xr-x  251 blyth  staff  8534 Nov 29  2017 GMeshLib
    drwxr-xr-x    8 blyth  staff   272 Nov 29  2017 GMergedMesh
    epsilon:1 blyth$ 

With the last one::

    epsilon:1 blyth$ ll /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/
    total 0
    drwxr-xr-x    3 blyth  staff    96 Apr  4 21:59 ..
    drwxr-xr-x    8 blyth  staff   256 Apr  4 21:59 GMergedMesh
    drwxr-xr-x    4 blyth  staff   128 Apr  4 21:59 MeshIndex
    drwxr-xr-x  251 blyth  staff  8032 Apr  4 21:59 GMeshLib
    drwxr-xr-x    3 blyth  staff    96 Apr  4 21:59 GMaterialLib
    drwxr-xr-x    5 blyth  staff   160 Apr  4 21:59 GSurfaceLib
    drwxr-xr-x    5 blyth  staff   160 Apr  4 21:59 GScintillatorLib
    drwxr-xr-x    3 blyth  staff    96 Apr  4 21:59 GSourceLib
    drwxr-xr-x    6 blyth  staff   192 Apr  4 21:59 GItemList
    drwxr-xr-x    5 blyth  staff   160 Apr  4 22:00 GBndLib
    drwxr-xr-x    2 blyth  staff    64 Apr  5 10:02 MeshIndexAnalytic
    drwxr-xr-x    5 blyth  staff   160 Jun 23 21:09 GNodeLib
    drwxr-xr-x   13 blyth  staff   416 Jun 23 21:15 .
    epsilon:1 blyth$ 

Make some fresh geocaches into slot 101 and 103 with analytic enabled::

    epsilon:optickscore blyth$ OPTICKS_RESOURCE_LAYOUT=101 OKTest -G --gltf 1
    epsilon:optickscore blyth$ OPTICKS_RESOURCE_LAYOUT=103 OKTest -G --gltf 3


Using gltf 1 vs 3 does make a difference to GMergedMeshAnalytic::

    epsilon:96ff965744a2f6b78c24e33c80d3a4cd blyth$ diff -r --brief 101 103
    Files 101/GMergedMeshAnalytic/0/bbox.npy and 103/GMergedMeshAnalytic/0/bbox.npy differ
    Files 101/GMergedMeshAnalytic/0/boundaries.npy and 103/GMergedMeshAnalytic/0/boundaries.npy differ
    Files 101/GMergedMeshAnalytic/0/center_extent.npy and 103/GMergedMeshAnalytic/0/center_extent.npy differ
    Files 101/GMergedMeshAnalytic/0/colors.npy and 103/GMergedMeshAnalytic/0/colors.npy differ
    ...
    Files 101/GMergedMeshAnalytic/5/sensors.npy and 103/GMergedMeshAnalytic/5/sensors.npy differ
    Files 101/GMergedMeshAnalytic/5/vertices.npy and 103/GMergedMeshAnalytic/5/vertices.npy differ

    Files 101/GMeshLib/24/transforms.npy and 103/GMeshLib/24/transforms.npy differ
    Files 101/GMeshLib/42/transforms.npy and 103/GMeshLib/42/transforms.npy differ
    epsilon:96ff965744a2f6b78c24e33c80d3a4cd blyth$ 


And GMeshLib transforms (which is unexpected).  Suspect an uninitialized transform
in some GMesh (perhaps placeholders) for meshes 24 and 42.

* :doc:`GMeshLib_uninitialized_transforms`




