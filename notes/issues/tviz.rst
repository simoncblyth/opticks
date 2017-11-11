tviz
======


*tviz-jpmt-scintillation* issue reported by Axel 

* a null PVName was attempted to be added to a GItemList 

* cause is that much of Opticks now assumes m_volnames true, 
  but that ctrl keys are configured in op.sh in OPTICKS_CTRL 
  and the volnames key was not present for jpmt geometry



Try to fix by adding the ctrl key::

    435 op-geometry-setup-juno()
    436 {
    437    local geo=${1:-JPMT}
    438    if [ "$geo" == "JUNO" ]; then
    439        export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JUNO
    440        export OPTICKS_QUERY="range:1:50000"
    441        export OPTICKS_CTRL="volnames"
    442    elif [ "$geo" == "JPMT" ]; then
    443        export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JPMT
    444        export OPTICKS_QUERY="range:1:289734"  # 289733+1 all test3.dae volumes
    445        export OPTICKS_CTRL="volnames"
    446    elif [ "$geo" == "JTST" ]; then
    447        export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_JTST
    448        export OPTICKS_QUERY="range:1:50000"
    449        export OPTICKS_CTRL="volnames"
    450    elif [ "$geo" == "J1707" ]; then
    451        export OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_J1707
    452        export OPTICKS_QUERY="all"
    453        export OPTICKS_CTRL="volnames"
    454    fi
    455 }




::

    delta:opticks blyth$ tviz-;tviz-jpmt-scintillation -D





::

    also tried to run: tviz-jpmt-scintillation. These are the results:

    gpu-CELSIUS-R940 opticks # tviz-jpmt-scintillation
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --scintillation
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OKTest
    288 -rwxr-xr-x 1 root root 292536 Nov 11 02:53 /usr/local/opticks/lib/OKTest
    proceeding.. : /usr/local/opticks/lib/OKTest --size 1920,1080,1 --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --scintillation
    2017-11-11 03:28:33.857 INFO  [30555] [Opticks::dumpArgs@816] Opticks::configure argc 14
      0 : /usr/local/opticks/lib/OKTest
      1 : --size
      2 : 1920,1080,1
      3 : --jpmt
      4 : --jwire
      5 : --target
      6 : 64670
      7 : --load
      8 : --animtimemax
      9 : 200
     10 : --timemax
     11 : 200
     12 : --optixviz
     13 : --scintillation
    2017-11-11 03:28:33.859 INFO  [30555] [OpticksHub::configure@234] OpticksHub::configure m_gltf 0
    2017-11-11 03:28:33.860 INFO  [30555] [OpticksHub::loadGeometry@364] OpticksHub::loadGeometry START
    2017-11-11 03:28:33.861 INFO  [30555] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-11 03:28:33.865 INFO  [30555] [OpticksGeometry::loadGeometry@102] OpticksGeometry::loadGeometry START 
    2017-11-11 03:28:33.865 INFO  [30555] [OpticksGeometry::loadGeometryBase@134] OpticksGeometry::loadGeometryBase START 
    2017-11-11 03:28:33.865 INFO  [30555] [GGeo::loadGeometry@532] GGeo::loadGeometry START loaded 0 gltf 0
    2017-11-11 03:28:33.865 INFO  [30555] [AssimpGGeo::load@135] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/juno/test3.dae query range:1:289734 ctrl  verbosity 0
    2017-11-11 03:28:33.865 INFO  [30555] [AssimpImporter::import@195] AssimpImporter::import path /usr/local/opticks/opticksdata/export/juno/test3.dae flags 32779
    2017-11-11 03:28:36.819 INFO  [30555] [AssimpImporter::Summary@112] AssimpImporter::import DONE
    2017-11-11 03:28:36.819 INFO  [30555] [AssimpImporter::Summary@113] AssimpImporter::info m_aiscene  NumMaterials 22 NumMeshes 27
    2017-11-11 03:28:38.214 INFO  [30555] [AssimpGGeo::load@150] AssimpGGeo::load select START 
    2017-11-11 03:28:38.214 INFO  [30555] [AssimpSelection::init@78] AssimpSelection::AssimpSelection before SelectNodes  queryType range query_string range:1:289734 query_name NULL query_index 0 query_depth 0 no_selection 0 nrange 2 : 1 : 289734
    2017-11-11 03:28:38.271 INFO  [30555] [AssimpSelection::init@85] AssimpSelection::AssimpSelection after SelectNodes  m_selection size 289732 out of m_count 289733
    2017-11-11 03:28:38.271 INFO  [30555] [AssimpSelection::findBounds@161] AssimpSelection::findBounds  NumSelected 289732
    2017-11-11 03:28:38.286 INFO  [30555] [AssimpGGeo::load@154] AssimpGGeo::load select DONE  
    2017-11-11 03:28:38.286 FATAL [30555] [NSensorList::read@133] NSensorList::read failed to open /usr/local/opticks/opticksdata/export/juno/test3.idmap
    2017-11-11 03:28:38.286 INFO  [30555] [OpticksResource::getSensorList@1115] OpticksResource::getSensorList NSensorList:  NSensor count 0 distinct identier count 0
    2017-11-11 03:28:38.286 INFO  [30555] [AssimpGGeo::convert@172] AssimpGGeo::convert ctrl 
    2017-11-11 03:28:38.286 INFO  [30555] [AssimpGGeo::convertMaterials@372] AssimpGGeo::convertMaterials  query  mNumMaterials 22
    2017-11-11 03:28:38.288 INFO  [30555] [GMaterialLib::addTestMaterials@832] GMaterialLib::addTestMaterials name                  GlassSchottF2 path $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy
    2017-11-11 03:28:38.289 INFO  [30555] [GMaterialLib::addTestMaterials@832] GMaterialLib::addTestMaterials name                    MainH2OHale path $OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy
    2017-11-11 03:28:38.424 WARN  [30555] [AssimpGGeo::convertSensors@533] AssimpGGeo::convertSensors m_cathode NULL : no material with an efficiency property ?  
    2017-11-11 03:28:38.424 INFO  [30555] [AssimpGGeo::convertMeshes@743] AssimpGGeo::convertMeshes NumMeshes 27
    2017-11-11 03:28:38.486 INFO  [30555] [AssimpGGeo::convertStructure@782] AssimpGGeo::convertStructure 
    2017-11-11 03:28:38.487 INFO  [30555] [GPropertyLib::getIndex@345] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Galactic]
    2017-11-11 03:28:38.487 INFO  [30555] [GPropertyLib::close@396] GPropertyLib::close type GMaterialLib buf 17,2,39,4
    2017-11-11 03:28:38.487 INFO  [30555] [GPropertyLib::getIndex@345] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-11-11 03:28:38.487 INFO  [30555] [GPropertyLib::close@396] GPropertyLib::close type GSurfaceLib buf 11,2,39,4
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_S_construct null not valid
    /home/gpu/opticks/bin/op.sh: line 755: 30555 Aborted                 /usr/local/opticks/lib/OKTest --size 1920,1080,1 --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --scintillation
    /home/gpu/opticks/bin/op.sh RC 134

    Is the test3.idmap a missing geometry file?



::

    delta:~ blyth$ tviz-jpmt-scintillation -D
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --scintillation -D
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OKTest
    288 -rwxr-xr-x  1 blyth  staff  145440 Nov 11 16:10 /usr/local/opticks/lib/OKTest
    proceeding.. : lldb /usr/local/opticks/lib/OKTest -- --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --scintillation -D
    (lldb) target create "/usr/local/opticks/lib/OKTest"

    ...

    (lldb) bt
    * thread #1: tid = 0x444094, 0x00007fff8aff2732 libsystem_c.dylib`strlen + 18, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00007fff8aff2732 libsystem_c.dylib`strlen + 18
        frame #1: 0x0000000102041a90 libGGeo.dylib`GItemList::add(char const*) [inlined] std::__1::char_traits<char>::length(__s=0x0000000000000000) + 208 at string:651
        frame #2: 0x0000000102041a6c libGGeo.dylib`GItemList::add(char const*) [inlined] std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::basic_string(this=0x00007fff5fbfbfa0, __s=0x0000000000000000) + 21 at string:1968
        frame #3: 0x0000000102041a57 libGGeo.dylib`GItemList::add(char const*) [inlined] std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::basic_string(this=0x00007fff5fbfbfa0, this=0x00007fff5fbfc008, __s=0x0000000000000000, __x=0x0000000147ea3398) + 77 at string:1972
        frame #4: 0x0000000102041a0a libGGeo.dylib`GItemList::add(this=0x0000000147ea3350, name=0x0000000000000000) + 74 at GItemList.cc:129
        frame #5: 0x00000001021aecf9 libGGeo.dylib`GNodeLib::add(this=0x0000000105d17e10, solid=0x0000000147ea6640) + 1001 at GNodeLib.cc:178
        frame #6: 0x000000010219ca04 libGGeo.dylib`GGeo::add(this=0x0000000105d13d70, solid=0x0000000147ea6640) + 36 at GGeo.cc:866
        frame #7: 0x0000000101e9c59b libAssimpRap.dylib`AssimpGGeo::convertStructure(this=0x00007fff5fbfc920, gg=0x0000000105d13d70, node=0x000000010b565c80, depth=0, parent=0x0000000000000000) + 187 at AssimpGGeo.cc:831
        frame #8: 0x0000000101e9996b libAssimpRap.dylib`AssimpGGeo::convertStructure(this=0x00007fff5fbfc920, gg=0x0000000105d13d70) + 299 at AssimpGGeo.cc:784
        frame #9: 0x0000000101e97580 libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007fff5fbfc920, ctrl=0x00007fff5fbfef5a) + 384 at AssimpGGeo.cc:179
        frame #10: 0x0000000101e97394 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x0000000105d13d70) + 1700 at AssimpGGeo.cc:164
        frame #11: 0x000000010219a3bb libGGeo.dylib`GGeo::loadFromG4DAE(this=0x0000000105d13d70) + 251 at GGeo.cc:579
        frame #12: 0x000000010219a010 libGGeo.dylib`GGeo::loadGeometry(this=0x0000000105d13d70) + 400 at GGeo.cc:539
        frame #13: 0x0000000102302742 libOpticksGeometry.dylib`OpticksGeometry::loadGeometryBase(this=0x0000000105d14260) + 1410 at OpticksGeometry.cc:156
        frame #14: 0x0000000102301e93 libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105d14260) + 243 at OpticksGeometry.cc:104
        frame #15: 0x0000000102306269 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105d0e830) + 409 at OpticksHub.cc:375
        frame #16: 0x0000000102305289 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105d0e830) + 137 at OpticksHub.cc:186
        frame #17: 0x0000000102305150 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0e830, ok=0x0000000105c21b40) + 464 at OpticksHub.cc:167
        frame #18: 0x00000001023053ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0e830, ok=0x0000000105c21b40) + 29 at OpticksHub.cc:169
        frame #19: 0x0000000103cab1b6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9c8, argc=13, argv=0x00007fff5fbfeaa0, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #20: 0x0000000103cab61b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9c8, argc=13, argv=0x00007fff5fbfeaa0, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #21: 0x000000010000b31d OKTest`main(argc=13, argv=0x00007fff5fbfeaa0) + 1373 at OKTest.cc:58
        frame #22: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 5
    frame #5: 0x00000001021aecf9 libGGeo.dylib`GNodeLib::add(this=0x0000000105d17e10, solid=0x0000000147ea6640) + 1001 at GNodeLib.cc:178
       175      if(!m_pvlist) m_pvlist = new GItemList("PVNames", m_reldir) ; 
       176      if(!m_lvlist) m_lvlist = new GItemList("LVNames", m_reldir) ; 
       177  
    -> 178      m_lvlist->add(solid->getLVName()); 
       179      m_pvlist->add(solid->getPVName()); 
       180  
       181      // NB added in tandem, so same counts and same index as the solids  
    (lldb) p solid
    (GSolid *) $2 = 0x0000000147ea6640
    (lldb) p solid->getLVName()
    (const char *) $3 = 0x0000000000000000
    (lldb) p solid->getPVName()
    (const char *) $4 = 0x0000000000000000
    (lldb) 


::

    delta:~ blyth$ opticks-find setPVName
    ./assimprap/AssimpGGeo.cc:        solid->setPVName(pv);
    ./ggeo/GMaker.cc:    solid->setPVName( strdup(pvn.c_str()) );
    ./ggeo/GScene.cc:    node->setPVName( pvname.c_str() );
    ./ggeo/GSolid.cc:void GSolid::setPVName(const char* pvname)
    ./ggeo/GSolid.hh:      void setPVName(const char* pvname);
    delta:opticks blyth$ 



::

    0104 void AssimpGGeo::init()
     105 {
    ...
     117 
     118     m_volnames = m_ggeo->isVolnames();



    1035     if(m_volnames)
    1036     {
    1037         solid->setPVName(pv);
    1038         solid->setLVName(lv);
    1047     }
    1048 


     349 void GGeo::init()
     350 {
     351    LOG(trace) << "GGeo::init" ;
     352 
     353    OpticksResource* resource = m_ok->getResource();
     354    const char* idpath = m_ok->getIdPath() ;
     355 
     ...
     376    const char* ctrl = resource->getCtrl() ;
     377 
     378    m_volnames = GGeo::ctrlHasKey(ctrl, "volnames");
     379 


