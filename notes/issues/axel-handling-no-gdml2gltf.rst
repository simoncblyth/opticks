axel-handling-no-gdml2gltf
============================


Report from Axel : tviz-jun-scintillation
---------------------------------------------

* presumably need to improve handling/error-reporting of missing analytic conversion
  because this works for me

* try to reproduce Axel error with some other geometry


::

    op --j1707 --gdml2dgltf 



::

    simon:issues blyth$ t tviz-jun-scintillation
    tviz-jun-scintillation () 
    { 
        tviz-jun- --scintillation $*
    }
    simon:issues blyth$ t tviz-jun-
    tviz-jun- () 
    { 
        op.sh --j1707 --gltf 3 --animtimemax 200 --timemax 200 --optixviz $*
    }
    simon:issues blyth$ 


::

    Good morning Simon,

    I just started tviz-jun-scintillation and received the following terminal output (this is only a extract):

    2017-11-14 02:29:33.836 INFO  [10261] [GGeoLib::loadConstituents@184] GGeoLib::loadConstituents loaded 0 ridx ()
    2017-11-14 02:29:33.836 WARN  [10261] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GNodeLibAnalytic/PVNames.txt
    2017-11-14 02:29:33.836 WARN  [10261] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GNodeLibAnalytic/LVNames.txt
    2017-11-14 02:29:33.836 WARN  [10261] [Index::load@420] Index::load FAILED to load index  idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae itemtype GItemIndex Source path /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/MeshIndexAnalytic/GItemIndexSource.json Local path /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/MeshIndexAnalytic/GItemIndexLocal.json
    2017-11-14 02:29:33.836 WARN  [10261] [GItemIndex::loadIndex@173] GItemIndex::loadIndex failed for  idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae reldir MeshIndexAnalytic override NULL
    2017-11-14 02:29:33.836 INFO  [10261] [GMeshLib::loadMeshes@214] idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae
    2017-11-14 02:29:33.840 INFO  [10261] [GGeo::loadAnalyticFromCache@649] GGeo::loadAnalyticFromCache DONE
    2017-11-14 02:29:33.861 INFO  [10261] [GGeo::loadGeometry@562] GGeo::loadGeometry DONE
    2017-11-14 02:29:33.861 INFO  [10261] [OpticksGeometry::loadGeometryBase@168] OpticksGeometry::loadGeometryBase DONE 
    2017-11-14 02:29:33.861 INFO  [10261] [OpticksGeometry::loadGeometry@127] OpticksGeometry::loadGeometry DONE 
    2017-11-14 02:29:33.861 INFO  [10261] [OpticksHub::loadGeometry@406] OpticksHub::loadGeometry NOT modifying geometry
    2017-11-14 02:29:33.861 FATAL [10261] [OpticksHub::registerGeometry@468] OpticksHub::registerGeometry
    2017-11-14 02:29:33.861 INFO  [10261] [OpticksHub::getGGeoBasePrimary@700] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 3 ggb GScene
    OKTest: /home/gpu/opticks/opticksgeo/OpticksHub.cc:470: void OpticksHub::registerGeometry(): Assertion `mm0' failed.
    /home/gpu/opticks/bin/op.sh: line 783: 10261 Aborted                 /usr/local/opticks/lib/OKTest --size 1920,1080,1 --j1707 --gltf 3 --animtimemax 200 --timemax 200 --optixviz --scintillation
    /home/gpu/opticks/bin/op.sh RC 134

    Looks like some geometry files are missing. Is this an issue with the installation?

    Axel





Try to reproduce with lxe geom
-------------------------------------------------------------------

* actually lxe not a good example as lacks .gdml so cannot do gdml2gltf 


Make OpticksQuery with a EMPTY string select all nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:issues blyth$ op --lxe --gltf 1 
    288 -rwxr-xr-x  1 blyth  staff  145440 Nov 13 20:55 /usr/local/opticks/lib/OKTest
    proceeding.. : /usr/local/opticks/lib/OKTest --lxe --gltf 1
    2017-11-14 10:45:23.263 INFO  [4718448] [Opticks::dumpArgs@816] Opticks::configure argc 4
      0 : /usr/local/opticks/lib/OKTest
      1 : --lxe
      2 : --gltf
      3 : 1
    2017-11-14 10:45:23.264 INFO  [4718448] [OpticksHub::configure@240] OpticksHub::configure m_gltf 1
    2017-11-14 10:45:23.264 WARN  [4718448] [BTree::loadTree@49] BTree.loadTree: can't find file /usr/local/opticks/opticksdata/export/other/ChromaMaterialMap.json
    2017-11-14 10:45:23.264 INFO  [4718448] [OpticksHub::loadGeometry@370] OpticksHub::loadGeometry START
    2017-11-14 10:45:23.274 INFO  [4718448] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-14 10:45:23.278 INFO  [4718448] [OpticksGeometry::loadGeometry@102] OpticksGeometry::loadGeometry START 
    2017-11-14 10:45:23.278 INFO  [4718448] [OpticksGeometry::loadGeometryBase@134] OpticksGeometry::loadGeometryBase START 
    2017-11-14 10:45:23.279 INFO  [4718448] [GGeo::loadGeometry@522] GGeo::loadGeometry START loaded 0 gltf 1
    2017-11-14 10:45:23.280 INFO  [4718448] [AssimpGGeo::load@134] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/LXe/g4_00.dae query  ctrl  verbosity 0
    2017-11-14 10:45:23.305 INFO  [4718448] [AssimpImporter::import@195] AssimpImporter::import path /usr/local/opticks/opticksdata/export/LXe/g4_00.dae flags 32779
    ColladaLoader::BuildMaterialsExtras BAD DATA REF  key FASTTIMECONSTANT val FASTTIMECONSTANT 
    ColladaLoader::BuildMaterialsExtras BAD DATA REF  key RESOLUTIONSCALE val RESOLUTIONSCALE 
    ColladaLoader::BuildMaterialsExtras BAD DATA REF  key SCINTILLATIONYIELD val SCINTILLATIONYIELD 
    ColladaLoader::BuildMaterialsExtras BAD DATA REF  key SLOWTIMECONSTANT val SLOWTIMECONSTANT 
    ColladaLoader::BuildMaterialsExtras BAD DATA REF  key YIELDRATIO val YIELDRATIO 
    2017-11-14 10:45:23.331 INFO  [4718448] [AssimpImporter::Summary@112] AssimpImporter::import DONE
    2017-11-14 10:45:23.332 INFO  [4718448] [AssimpImporter::Summary@113] AssimpImporter::info m_aiscene  NumMaterials 6 NumMeshes 6
    2017-11-14 10:45:23.332 INFO  [4718448] [AssimpGGeo::load@149] AssimpGGeo::load select START 
    2017-11-14 10:45:23.332 INFO  [4718448] [AssimpSelection::init@78] AssimpSelection::AssimpSelection before SelectNodes  queryType undefined query_string  query_name NULL query_index 0 query_depth 0 no_selection 0
    2017-11-14 10:45:23.333 INFO  [4718448] [AssimpSelection::init@85] AssimpSelection::AssimpSelection after SelectNodes  m_selection size 0 out of m_count 68
    Assertion failed: (m_selection.size() > 0), function init, file /Users/blyth/opticks/assimprap/AssimpSelection.cc, line 91.
    /Users/blyth/opticks/bin/op.sh: line 783: 71653 Abort trap: 6           /usr/local/opticks/lib/OKTest --lxe --gltf 1
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:issues blyth$ 
    (lldb) bt
    * thread #1: tid = 0x480514, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101e8e387 libAssimpRap.dylib`AssimpSelection::init(this=0x0000000105c2f420) + 1127 at AssimpSelection.cc:91
        frame #5: 0x0000000101e8dede libAssimpRap.dylib`AssimpSelection::AssimpSelection(this=0x0000000105c2f420, root=0x0000000105c33c20, query=0x0000000105d0d540) + 222 at AssimpSelection.cc:34
        frame #6: 0x0000000101e8e3d5 libAssimpRap.dylib`AssimpSelection::AssimpSelection(this=0x0000000105c2f420, root=0x0000000105c33c20, query=0x0000000105d0d540) + 37 at AssimpSelection.cc:35
        frame #7: 0x0000000101ea04e5 libAssimpRap.dylib`AssimpImporter::select(this=0x00007fff5fbfcd18, query=0x0000000105d0d540) + 133 at AssimpImporter.cc:240
        frame #8: 0x0000000101e971f7 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x0000000105c22870) + 1303 at AssimpGGeo.cc:151
        frame #9: 0x000000010219b31b libGGeo.dylib`GGeo::loadFromG4DAE(this=0x0000000105c22870) + 251 at GGeo.cc:569
        frame #10: 0x000000010219af70 libGGeo.dylib`GGeo::loadGeometry(this=0x0000000105c22870) + 400 at GGeo.cc:529
        frame #11: 0x0000000102303742 libOpticksGeometry.dylib`OpticksGeometry::loadGeometryBase(this=0x0000000105c227e0) + 1410 at OpticksGeometry.cc:156
        frame #12: 0x0000000102302e93 libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105c227e0) + 243 at OpticksGeometry.cc:104
        frame #13: 0x0000000102307269 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105d0df90) + 409 at OpticksHub.cc:375
        frame #14: 0x0000000102306289 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105d0df90) + 137 at OpticksHub.cc:186
        frame #15: 0x0000000102306150 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0df90, ok=0x0000000105c21950) + 464 at OpticksHub.cc:167
        frame #16: 0x00000001023063ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0df90, ok=0x0000000105c21950) + 29 at OpticksHub.cc:169
        frame #17: 0x0000000103cac1b6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfeab8, argc=5, argv=0x00007fff5fbfeb98, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #18: 0x0000000103cac61b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfeab8, argc=5, argv=0x00007fff5fbfeb98, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #19: 0x000000010000b31d OKTest`main(argc=5, argv=0x00007fff5fbfeb98) + 1373 at OKTest.cc:58
        frame #20: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #21: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 8
    frame #8: 0x0000000101e971f7 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x0000000105c22870) + 1303 at AssimpGGeo.cc:151
       148  
       149      LOG(info) << "AssimpGGeo::load select START " ; 
       150  
    -> 151      AssimpSelection* selection = assimp.select(query);
       152  
       153      LOG(info) << "AssimpGGeo::load select DONE  " ; 
       154  
    (lldb) p query 
    (OpticksQuery *) $0 = 0x0000000105d0d540
    (lldb) p query->m_query_string
    (const char *) $1 = 0x0000000105d0e9f0 ""



OPTICKS_CTRL volnames should default to true ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    simon:issues blyth$ op --lxe --gltf 1 


    Assertion failed: (name), function add, file /Users/blyth/opticks/ggeo/GItemList.cc, line 129.
    Process 72907 stopped
    * thread #1: tid = 0x481291, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x481291, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001020427fa libGGeo.dylib`GItemList::add(this=0x0000000105d18250, name=0x0000000000000000) + 106 at GItemList.cc:129
        frame #5: 0x00000001021afc59 libGGeo.dylib`GNodeLib::add(this=0x0000000105e04590, solid=0x0000000105d16350) + 1001 at GNodeLib.cc:178
        frame #6: 0x000000010219d964 libGGeo.dylib`GGeo::add(this=0x0000000105e00840, solid=0x0000000105d16350) + 36 at GGeo.cc:851
        frame #7: 0x0000000101e9c58b libAssimpRap.dylib`AssimpGGeo::convertStructure(this=0x00007fff5fbfca10, gg=0x0000000105e00840, node=0x0000000105e11bf0, depth=0, parent=0x0000000000000000) + 187 at AssimpGGeo.cc:830
        frame #8: 0x0000000101e9995b libAssimpRap.dylib`AssimpGGeo::convertStructure(this=0x00007fff5fbfca10, gg=0x0000000105e00840) + 299 at AssimpGGeo.cc:783
        frame #9: 0x0000000101e97570 libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007fff5fbfca10, ctrl=0x00007fff5fbfefb9) + 384 at AssimpGGeo.cc:178
        frame #10: 0x0000000101e97384 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x0000000105e00840) + 1700 at AssimpGGeo.cc:163
        frame #11: 0x000000010219b31b libGGeo.dylib`GGeo::loadFromG4DAE(this=0x0000000105e00840) + 251 at GGeo.cc:569
        frame #12: 0x000000010219af70 libGGeo.dylib`GGeo::loadGeometry(this=0x0000000105e00840) + 400 at GGeo.cc:529
        frame #13: 0x0000000102303742 libOpticksGeometry.dylib`OpticksGeometry::loadGeometryBase(this=0x0000000105e007b0) + 1410 at OpticksGeometry.cc:156
        frame #14: 0x0000000102302e93 libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105e007b0) + 243 at OpticksGeometry.cc:104
        frame #15: 0x0000000102307269 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105d0df90) + 409 at OpticksHub.cc:375
        frame #16: 0x0000000102306289 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105d0df90) + 137 at OpticksHub.cc:186
        frame #17: 0x0000000102306150 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0df90, ok=0x0000000105c21950) + 464 at OpticksHub.cc:167
        frame #18: 0x00000001023063ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0df90, ok=0x0000000105c21950) + 29 at OpticksHub.cc:169
        frame #19: 0x0000000103cac1b6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfeab8, argc=5, argv=0x00007fff5fbfeb98, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #20: 0x0000000103cac61b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfeab8, argc=5, argv=0x00007fff5fbfeb98, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #21: 0x000000010000b31d OKTest`main(argc=5, argv=0x00007fff5fbfeb98) + 1373 at OKTest.cc:58
        frame #22: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #23: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) f 7
    frame #7: 0x0000000101e9c58b libAssimpRap.dylib`AssimpGGeo::convertStructure(this=0x00007fff5fbfca10, gg=0x0000000105e00840, node=0x0000000105e11bf0, depth=0, parent=0x0000000000000000) + 187 at AssimpGGeo.cc:830
       827  
       828      solid->setSelected(selected);
       829  
    -> 830      gg->add(solid);
       831  
       832      if(parent) // GNode hookup
       833      {
    (lldb) p solid
    (GSolid *) $0 = 0x0000000105d16350
    (lldb) f 6
    frame #6: 0x000000010219d964 libGGeo.dylib`GGeo::add(this=0x0000000105e00840, solid=0x0000000105d16350) + 36 at GGeo.cc:851
       848  }
       849  void GGeo::add(GSolid* solid)
       850  {
    -> 851      m_nodelib->add(solid);
       852  }
       853  GSolid* GGeo::getSolid(unsigned index)
       854  {
    (lldb) f 5
    frame #5: 0x00000001021afc59 libGGeo.dylib`GNodeLib::add(this=0x0000000105e04590, solid=0x0000000105d16350) + 1001 at GNodeLib.cc:178
       175      if(!m_pvlist) m_pvlist = new GItemList("PVNames", m_reldir) ; 
       176      if(!m_lvlist) m_lvlist = new GItemList("LVNames", m_reldir) ; 
       177  
    -> 178      m_lvlist->add(solid->getLVName()); 
       179      m_pvlist->add(solid->getPVName()); 
       180  
       181      // NB added in tandem, so same counts and same index as the solids  
    (lldb) 




flip volnames default to ON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Switch to volnames ON by default and novolnames key switching it off.

Before::

    simon:optickscore blyth$ opticks-find volnames
    ./bin/op.sh:    export OPTICKS_CTRL="volnames"
    ./bin/op.sh:       export OPTICKS_CTRL="volnames"
    ./bin/op.sh:       export OPTICKS_CTRL="volnames"
    ./bin/op.sh:       export OPTICKS_CTRL="volnames"
    ./bin/op.sh:       export OPTICKS_CTRL="volnames"
    ./ok/ggv.sh:   export OPTICKS_CTRL="volnames"
    ./assimprap/assimprap.bash:    OpticksResource::readEnvironment USING DEFAULT geo ctrl volnames
    ./cfg4/cfg4.bash:    ctrl     : volnames
    ./externals/assimp.bash:    2017-07-03 11:53:06.260 INFO  [2788361] [AssimpGGeo::load@131] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:4448:4456 ctrl volnames verbosity 0
    ./assimprap/AssimpGGeo.cc:   m_volnames(m_ok->hasCtrlKey("volnames")),
    ./assimprap/AssimpGGeo.cc:    return m_volnames ; 
    ./assimprap/AssimpGGeo.cc:    if(m_volnames)
    ./ggeo/GBndLib.cc:    // hmm: when need to create surf, need the volnames ?
    ./optickscore/OpticksResource.cc:const char* OpticksResource::DEFAULT_CTRL = "volnames" ; 
    ./assimprap/AssimpGGeo.hh:    bool             m_volnames ; 
    simon:opticks blyth$ 




succeed to reproduce the error reporting is pretty good
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* TODO: somewhere very early (maybe Opticks::configure) check .dae .gdml .gltf file existance relative to options

::

    op --lxe --gltf 1 

    ...

    2017-11-14 11:28:45.937 INFO  [4742854] [NMeta::write@187] write to /usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GSurfaceLib/GPropertyLibMetadata.json
    2017-11-14 11:28:45.937 INFO  [4742854] [*GScintillatorLib::createBuffer@109] GScintillatorLib::createBuffer  ni 0 nj 4096 nk 1
    2017-11-14 11:28:45.937 INFO  [4742854] [GPropertyLib::close@396] GPropertyLib::close type GScintillatorLib buf 0,4096,1
    2017-11-14 11:28:45.937 INFO  [4742854] [NPY<float>::save@635] NPYBase::save creating directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GScintillatorLib]/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GScintillatorLib/GScintillatorLib.npy
    2017-11-14 11:28:45.938 INFO  [4742854] [NPY<float>::save@638] NPYBase::save created directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GScintillatorLib]
    2017-11-14 11:28:45.938 FATAL [4742854] [NPY<float>::save@658] NPY values NULL, SKIP attempt to save   itemcount 0 itemshape 4096,1 native /usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GScintillatorLib/GScintillatorLib.npy
    2017-11-14 11:28:45.938 INFO  [4742854] [*GSourceLib::createBuffer@95] GSourceLib::createBuffer adding standard source 
    2017-11-14 11:28:45.938 INFO  [4742854] [GPropertyLib::close@396] GPropertyLib::close type GSourceLib buf 1,1024,1
    2017-11-14 11:28:45.938 INFO  [4742854] [NPY<float>::save@635] NPYBase::save creating directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GSourceLib]/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GSourceLib/GSourceLib.npy
    2017-11-14 11:28:45.938 INFO  [4742854] [NPY<float>::save@638] NPYBase::save created directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GSourceLib]
    2017-11-14 11:28:45.938 INFO  [4742854] [int>::save@635] NPYBase::save creating directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GBndLib]/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GBndLib/GBndLibIndex.npy
    2017-11-14 11:28:45.938 INFO  [4742854] [int>::save@638] NPYBase::save created directories [/usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GBndLib]
    2017-11-14 11:28:45.939 INFO  [4742854] [GGeo::loadAnalyticFromGLTF@585] GGeo::loadAnalyticFromGLTF START
    2017-11-14 11:28:45.939 FATAL [4742854] [*NScene::Load@127] NScene:Load MISSING PATH gltfbase /usr/local/opticks/opticksdata/export/LXe gltfname g4_00.gltf gltfconfig 0x7fa0ba604650
    2017-11-14 11:28:45.939 FATAL [4742854] [GScene::initFromGLTF@172] NScene::Load FAILED
    2017-11-14 11:28:45.939 INFO  [4742854] [GGeo::loadAnalyticFromGLTF@596] GGeo::loadAnalyticFromGLTF DONE
    2017-11-14 11:28:45.939 INFO  [4742854] [GGeo::saveAnalytic@617] GGeo::saveAnalytic
    2017-11-14 11:28:45.939 INFO  [4742854] [GGeoLib::dump@300] GScene::save
    2017-11-14 11:28:45.939 INFO  [4742854] [GGeoLib::dump@301] GGeoLib ANALYTIC  numMergedMesh 0 ptr 0x7fa0ba66c250
     num_total_volumes 0 num_instanced_volumes 0 num_global_volumes 0
    2017-11-14 11:28:45.939 WARN  [4742854] [GNodeLib::save@64] GNodeLib::save pvlist NULL 
    2017-11-14 11:28:45.939 WARN  [4742854] [GNodeLib::save@74] GNodeLib::save lvlist NULL 
    2017-11-14 11:28:45.939 ERROR [4742854] [GTreePresent::traverse@35] GTreePresent::traverse top NULL 
    2017-11-14 11:28:45.939 INFO  [4742854] [GTreePresent::write@108] GTreePresent::write /usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GNodeLibAnalytic/GTreePresent.txt
    2017-11-14 11:28:45.939 INFO  [4742854] [GTreePresent::write@113] GTreePresent::write /usr/local/opticks/opticksdata/export/LXe/g4_00.d41d8cd98f00b204e9800998ecf8427e.dae/GNodeLibAnalytic/GTreePresent.txtDONE
    2017-11-14 11:28:45.939 WARN  [4742854] [GMeshLib::save@72] GMeshLib::save m_meshindex NULL 
    2017-11-14 11:28:45.944 INFO  [4742854] [GGeo::loadGeometry@562] GGeo::loadGeometry DONE
    2017-11-14 11:28:45.944 INFO  [4742854] [OpticksGeometry::loadGeometryBase@168] OpticksGeometry::loadGeometryBase DONE 
    2017-11-14 11:28:45.944 INFO  [4742854] [OpticksGeometry::fixGeometry@180] OpticksGeometry::fixGeometry
    2017-11-14 11:28:45.944 INFO  [4742854] [MFixer::fixMesh@37] MFixer::fixMesh NumSolids 68 NumMeshes 6
    2017-11-14 11:28:45.945 INFO  [4742854] [OpticksGeometry::loadGeometry@127] OpticksGeometry::loadGeometry DONE 
    2017-11-14 11:28:45.945 INFO  [4742854] [OpticksHub::loadGeometry@406] OpticksHub::loadGeometry NOT modifying geometry
    2017-11-14 11:28:45.945 FATAL [4742854] [OpticksHub::registerGeometry@468] OpticksHub::registerGeometry
    2017-11-14 11:28:45.945 INFO  [4742854] [*OpticksHub::getGGeoBasePrimary@700] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 1 ggb GScene
    Assertion failed: (mm0), function registerGeometry, file /Users/blyth/opticks/opticksgeo/OpticksHub.cc, line 470.
    /Users/blyth/opticks/bin/op.sh: line 786: 84492 Abort trap: 6           /usr/local/opticks/lib/OKTest --lxe --gltf 1
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:issues blyth$ 
    simon:issues blyth$ 
    simon:issues blyth$ 
    simon:issues blyth$ 


