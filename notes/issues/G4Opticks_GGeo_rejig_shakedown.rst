G4Opticks/GGeo rejig collateral damages
===========================================

Next: 

* :doc:`tboolean-box-analysis-fail`


Issues
----------

Enabling G4Opticks to run from cache has some knock on issues to fix 
when not running from cache, with eg::

    epsilon:opticks blyth$ opticksaux-;G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 



FIXED : Issue 1 : Calling GVolume::getIdentity whilst boundary unset(-1) asserts. : Fixed by reordering in X4PhysicalVolume::convertNode
-----------------------------------------------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ opticksaux-
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    (lldb) target create "G4OKTest"


    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff58ac8b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff58c93080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff58a241ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001090c022e libSysRap.dylib`SPack::Encode22(a=43, b=4294967295) at SPack.cc:58
        frame #5: 0x0000000107e638d9 libOpticksCore.dylib`OpticksShape::Encode(meshIndex=43, boundaryIndex=4294967295) at OpticksShape.cc:12
        frame #6: 0x00000001078c6405 libGGeo.dylib`GVolume::getShapeIdentity(this=0x000000011a78ad90) const at GVolume.cc:286
        frame #7: 0x00000001078c63a1 libGGeo.dylib`GVolume::getIdentity(this=0x000000011a78ad90) const at GVolume.cc:261
        frame #8: 0x0000000107917710 libGGeo.dylib`GNodeLib::addSensorVolume(this=0x0000000115365690, volume=0x000000011a78ad90) at GNodeLib.cc:338
        frame #9: 0x000000010790d884 libGGeo.dylib`GGeo::addSensorVolume(this=0x000000011535fb70, volume=0x000000011a78ad90) at GGeo.cc:974
        frame #10: 0x00000001003dd107 libExtG4.dylib`X4PhysicalVolume::convertNode(this=0x00007ffeefbfcf40, pv=0x000000010f8e7e30, parent=0x000000011a78a550, depth=13, pv_p=0x000000010f8e8240, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1370
        frame #11: 0x00000001003dc1ed libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x000000010f8e7e30, parent=0x000000011a78a550, depth=13, parent_pv=0x000000010f8e8240, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1012
        frame #12: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x000000010f8e8240, parent=0x000000011a789e10, depth=12, parent_pv=0x000000010f8ec050, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #13: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x000000010f8ec050, parent=0x000000011a771210, depth=11, parent_pv=0x0000000116c198c0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #14: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116c198c0, parent=0x000000011a770680, depth=10, parent_pv=0x0000000116c275f0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #15: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116c275f0, parent=0x000000011a76fa90, depth=9, parent_pv=0x0000000116c299f0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #16: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116c299f0, parent=0x000000011a76f320, depth=8, parent_pv=0x0000000116b424e0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #17: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116b424e0, parent=0x000000011a76e680, depth=7, parent_pv=0x0000000116b44c60, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #18: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116b44c60, parent=0x000000011a76df10, depth=6, parent_pv=0x0000000116e52190, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #19: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116e52190, parent=0x000000011a76d490, depth=5, parent_pv=0x0000000116e52ed0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #20: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116e52ed0, parent=0x000000011a76c9c0, depth=4, parent_pv=0x0000000116e544f0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #21: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116e544f0, parent=0x000000011a76c220, depth=3, parent_pv=0x0000000116e55290, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #22: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116e55290, parent=0x000000011695a400, depth=2, parent_pv=0x0000000116e552e0, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #23: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000116e552e0, parent=0x0000000116959d50, depth=1, parent_pv=0x0000000115964380, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #24: 0x00000001003dc24e libExtG4.dylib`X4PhysicalVolume::convertStructure_r(this=0x00007ffeefbfcf40, pv=0x0000000115964380, parent=0x0000000000000000, depth=0, parent_pv=0x0000000000000000, recursive_select=0x00007ffeefbfbdc3) at X4PhysicalVolume.cc:1027
        frame #25: 0x00000001003d693c libExtG4.dylib`X4PhysicalVolume::convertStructure(this=0x00007ffeefbfcf40) at X4PhysicalVolume.cc:947
        frame #26: 0x00000001003d58c3 libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfcf40) at X4PhysicalVolume.cc:201
        frame #27: 0x00000001003d5585 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfcf40, ggeo=0x000000011535fb70, top=0x0000000115964380) at X4PhysicalVolume.cc:180
        frame #28: 0x00000001003d4845 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfcf40, ggeo=0x000000011535fb70, top=0x0000000115964380) at X4PhysicalVolume.cc:171
        frame #29: 0x00000001000e5306 libG4OK.dylib`G4Opticks::translateGeometry(this=0x000000010f861580, top=0x0000000115964380) at G4Opticks.cc:663
        frame #30: 0x00000001000e49e4 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f861580, world=0x0000000115964380) at G4Opticks.cc:325
        frame #31: 0x00000001000e47f5 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f861580, gdmlpath="/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml") at G4Opticks.cc:310
        frame #32: 0x00000001000120c4 G4OKTest`G4OKTest::init(this=0x00007ffeefbfe908) at G4OKTest.cc:95
        frame #33: 0x0000000100012040 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe908, argc=3, argv=0x00007ffeefbfe950) at G4OKTest.cc:65
        frame #34: 0x0000000100012c13 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe908, argc=3, argv=0x00007ffeefbfe950) at G4OKTest.cc:64
        frame #35: 0x00000001000131c9 G4OKTest`main(argc=3, argv=0x00007ffeefbfe950) at G4OKTest.cc:147
        frame #36: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 8
    frame #8: 0x0000000107917710 libGGeo.dylib`GNodeLib::addSensorVolume(this=0x0000000115365690, volume=0x000000011a78ad90) at GNodeLib.cc:338
       335 	    unsigned sensorIndex = m_sensor_volumes.size() ;  
       336 	    m_sensor_volumes.push_back(volume); 
       337 	
    -> 338 	    glm::uvec4 id = volume->getIdentity();  
       339 	    m_sensor_identity.push_back(id); 
       340 	    m_num_sensors += 1 ; 
       341 	    return sensorIndex ; 
    (lldb) o sensorIndex
    error: 'o' is not a valid command.
    error: Unrecognized command 'o'.
    (lldb) p sensorIndex
    (unsigned int) $1 = 0
    (lldb) f 7
    frame #7: 0x00000001078c63a1 libGGeo.dylib`GVolume::getIdentity(this=0x000000011a78ad90) const at GVolume.cc:261
       258 	
       259 	glm::uvec4 GVolume::getIdentity() const 
       260 	{
    -> 261 	    glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ; 
       262 	    return id ; 
       263 	}
       264 	
    (lldb) f 6
    frame #6: 0x00000001078c6405 libGGeo.dylib`GVolume::getShapeIdentity(this=0x000000011a78ad90) const at GVolume.cc:286
       283 	
       284 	unsigned GVolume::getShapeIdentity() const
       285 	{
    -> 286 	    return OpticksShape::Encode( getMeshIndex(), getBoundary() ); 
       287 	}
       288 	
       289 	
    (lldb) f 5
    frame #5: 0x0000000107e638d9 libOpticksCore.dylib`OpticksShape::Encode(meshIndex=43, boundaryIndex=4294967295) at OpticksShape.cc:12
       9   	
       10  	unsigned OpticksShape::Encode( unsigned meshIndex, unsigned boundaryIndex )
       11  	{
    -> 12  	    return SPack::Encode22( meshIndex, boundaryIndex );
       13  	}
       14  	
       15  	unsigned OpticksShape::MeshIndex(const glm::uvec4& identity)
    (lldb) f 4
    frame #4: 0x00000001090c022e libSysRap.dylib`SPack::Encode22(a=43, b=4294967295) at SPack.cc:58
       55  	{
       56  	    assert( sizeof(unsigned) == 4 ); 
       57  	    assert( (a & 0xffff0000) == 0 ); 
    -> 58  	    assert( (b & 0xffff0000) == 0 ); 
       59  	    unsigned value = ( a << 16 ) | ( b << 0 ) ; 
       60  	    return value  ; 
       61  	}
    (lldb) 





FIXED in convertNode : Issue 2 : missed sensor_indices, must setSensorIndex for all volumes, -1 when not sensor
-----------------------------------------------------------------------------------------------------------------

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff58ac8b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff58c93080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff58a241ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001078ff8c3 libGGeo.dylib`GMergedMesh::mergeVolumeFaces(this=0x000000011dcae110, nface=12, faces=0x000000010fd0e560, node_indices=0x0000000115fe43f0, boundary_indices=0x0000000115fe4980, sensor_indices=0x0000000000000000) at GMergedMesh.cc:891
        frame #5: 0x00000001078fd968 libGGeo.dylib`GMergedMesh::mergeVolume(this=0x000000011dcae110, volume=0x0000000115fe47f0, selected=true) at GMergedMesh.cc:605
        frame #6: 0x00000001078fe65d libGGeo.dylib`GMergedMesh::traverse_r(this=0x000000011dcae110, node=0x0000000115fe47f0, depth=0, pass=1) at GMergedMesh.cc:393
        frame #7: 0x00000001078fe07d libGGeo.dylib`GMergedMesh::Create(ridx=0, base=0x0000000000000000, root=0x0000000115fe47f0) at GMergedMesh.cc:312
        frame #8: 0x00000001078cb4a2 libGGeo.dylib`GGeoLib::makeMergedMesh(this=0x0000000116b0fd80, index=0, base=0x0000000000000000, root=0x0000000115fe47f0) at GGeoLib.cc:294
        frame #9: 0x00000001078e4b36 libGGeo.dylib`GInstancer::makeMergedMeshAndInstancedBuffers(this=0x0000000116b10d10, verbosity=0) at GInstancer.cc:778
        frame #10: 0x00000001078e3b91 libGGeo.dylib`GInstancer::createInstancedMergedMeshes(this=0x0000000116b10d10, delta=true, verbosity=0) at GInstancer.cc:135
        frame #11: 0x000000010790bbfa libGGeo.dylib`GGeo::prepareVolumes(this=0x0000000116b6be00) at GGeo.cc:1257
        frame #12: 0x000000010790a8d6 libGGeo.dylib`GGeo::prepare(this=0x0000000116b6be00) at GGeo.cc:579
        frame #13: 0x0000000107909fb1 libGGeo.dylib`GGeo::postDirectTranslation(this=0x0000000116b6be00) at GGeo.cc:510
        frame #14: 0x00000001000e553c libG4OK.dylib`G4Opticks::translateGeometry(this=0x000000010f95e840, top=0x0000000116800040) at G4Opticks.cc:667
        frame #15: 0x00000001000e49e4 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f95e840, world=0x0000000116800040) at G4Opticks.cc:325
        frame #16: 0x00000001000e47f5 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f95e840, gdmlpath="/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml") at G4Opticks.cc:310
        frame #17: 0x00000001000120c4 G4OKTest`G4OKTest::init(this=0x00007ffeefbfe8d8) at G4OKTest.cc:95
        frame #18: 0x0000000100012040 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe8d8, argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:65
        frame #19: 0x0000000100012c13 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe8d8, argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:64
        frame #20: 0x00000001000131c9 G4OKTest`main(argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:147
        frame #21: 0x00007fff58978015 libdyld.dylib`start + 1
        frame #22: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) 


    lldb) f 4
    frame #4: 0x00000001078ff8c3 libGGeo.dylib`GMergedMesh::mergeVolumeFaces(this=0x000000011dcae110, nface=12, faces=0x000000010fd0e560, node_indices=0x0000000115fe43f0, boundary_indices=0x0000000115fe4980, sensor_indices=0x0000000000000000) at GMergedMesh.cc:891
       888 	{
       889 	    assert(node_indices);
       890 	    assert(boundary_indices);
    -> 891 	    assert(sensor_indices);
       892 	
       893 	    for(unsigned i=0 ; i < nface ; ++i )
       894 	    {
    (lldb) 


FIXED Issue 3 :  missing GParts for live running
--------------------------------------------------

Fixed by doing deferredCreateGParts from GGeo::postDirectTranslation::


     535 void GGeo::postDirectTranslation()
     536 {
     537     LOG(LEVEL) << "[" ;
     538 
     539     prepare();     // instances are formed here     
     540 
     541     LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ;
     542     GBndLib* blib = getBndLib();
     543     blib->fillMaterialLineMap();
     544     LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ;
     545 
     546     LOG(LEVEL) << "( GGeo::save " ;
     547     save();
     548     LOG(LEVEL) << ") GGeo::save " ;
     549 
     550 
     551     deferredCreateGParts();
     552 
     553     postDirectTranslationDump();
     554 
     555     LOG(LEVEL) << "]" ;
     556 }

::


    epsilon:g4ok blyth$ G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    ...
     num_total_volumes 4486 num_instanced_volumes 7744 num_global_volumes 4294964038 num_total_faces 483996 num_total_faces_woi 2533452 (woi:without instancing) 
       0 pts Y  GPts.NumPt  4486 lvIdx ( 248 247 21 0 7 6 3 2 3 2 ... 237 238 239 240 241 242 243 244 245)
       1 pts Y  GPts.NumPt     1 lvIdx ( 1)
       2 pts Y  GPts.NumPt     1 lvIdx ( 197)
       3 pts Y  GPts.NumPt     1 lvIdx ( 198)
       4 pts Y  GPts.NumPt     1 lvIdx ( 195)
       5 pts Y  GPts.NumPt     5 lvIdx ( 47 46 43 44 45)
    2020-10-15 16:29:10.593 INFO  [9825209] [OGeo::convert@263] [ nmm 6
    Assertion failed: (pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "), function makeAnalyticGeometry, file /Users/blyth/opticks/optixrap/OGeo.cc, line 683.
        frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106487a20 libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x0000000125723450, mm=0x000000011e269c90) at OGeo.cc:683
        frame #5: 0x0000000106485551 libOptiXRap.dylib`OGeo::makeOGeometry(this=0x0000000125723450, mergedmesh=0x000000011e269c90) at OGeo.cc:617
        frame #6: 0x0000000106483cf5 libOptiXRap.dylib`OGeo::makeGlobalGeometryGroup(this=0x0000000125723450, mm=0x000000011e269c90) at OGeo.cc:323
        frame #7: 0x0000000106482db9 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x0000000125723450, i=0) at OGeo.cc:303
        frame #8: 0x00000001064826fd libOptiXRap.dylib`OGeo::convert(this=0x0000000125723450) at OGeo.cc:269
        frame #9: 0x0000000106478649 libOptiXRap.dylib`OScene::init(this=0x000000012300c2f0) at OScene.cc:169
        frame #10: 0x0000000106477a01 libOptiXRap.dylib`OScene::OScene(this=0x000000012300c2f0, hub=0x0000000123007c70, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:91
        frame #11: 0x0000000106478c1d libOptiXRap.dylib`OScene::OScene(this=0x000000012300c2f0, hub=0x0000000123007c70, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:90
        frame #12: 0x0000000106388c16 libOKOP.dylib`OpEngine::OpEngine(this=0x000000012300c1f0, hub=0x0000000123007c70) at OpEngine.cc:75
        frame #13: 0x000000010638930d libOKOP.dylib`OpEngine::OpEngine(this=0x000000012300c1f0, hub=0x0000000123007c70) at OpEngine.cc:83
        frame #14: 0x000000010638ba16 libOKOP.dylib`OpPropagator::OpPropagator(this=0x000000012300bda0, hub=0x0000000123007c70, idx=0x000000012300beb0) at OpPropagator.cc:50
        frame #15: 0x000000010638bb15 libOKOP.dylib`OpPropagator::OpPropagator(this=0x000000012300bda0, hub=0x0000000123007c70, idx=0x000000012300beb0) at OpPropagator.cc:53
        frame #16: 0x000000010638a5f6 libOKOP.dylib`OpMgr::OpMgr(this=0x0000000123007c10, ok=0x000000010fb94e40) at OpMgr.cc:60
        frame #17: 0x000000010638a83d libOKOP.dylib`OpMgr::OpMgr(this=0x0000000123007c10, ok=0x000000010fb94e40) at OpMgr.cc:62
        frame #18: 0x00000001000e5c2c libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f8605b0, ggeo=0x0000000116334040) at G4Opticks.cc:397
        frame #19: 0x00000001000e4b01 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f8605b0, world=0x000000010fb011f0) at G4Opticks.cc:335
        frame #20: 0x00000001000e47f5 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010f8605b0, gdmlpath="/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml") at G4Opticks.cc:310
        frame #21: 0x00000001000120c4 G4OKTest`G4OKTest::init(this=0x00007ffeefbfe8d8) at G4OKTest.cc:95
        frame #22: 0x0000000100012040 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe8d8, argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:65
        frame #23: 0x0000000100012c13 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe8d8, argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:64
        frame #24: 0x00000001000131c9 G4OKTest`main(argc=3, argv=0x00007ffeefbfe928) at G4OKTest.cc:147
        frame #25: 0x00007fff58978015 libdyld.dylib`start + 1
        frame #26: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) 



FIXED : Issue 4 : live running giving sensor_identifier zeros (because GDML PV tree copyNo zero?) and standins zero( because collected too soon)
---------------------------------------------------------------------------------------------------------------------------------------------------

Hmm : this is a problem with the early collection of sensor identities before 
the GInstancer has defined them and labelled the tree.

DONE: move collection of sensor identities after tree labelling. 

Fixed by moving sensor identity collection later, to GInstancer::collectNodes_r.


::

    epsilon:g4ok blyth$ G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    ...
    2020-10-15 16:57:09.768 INFO  [9885824] [G4OKTest::init@103] [ setSensorData num_sensor 672 Geometry LIVE TRANSLATED
     sensor_index(dec)     0 (hex)     0 sensor_identifier(hex)       0
     sensor_index(dec)     1 (hex)     1 sensor_identifier(hex)       0
     sensor_index(dec)     2 (hex)     2 sensor_identifier(hex)       0
     sensor_index(dec)     3 (hex)     3 sensor_identifier(hex)       0
     sensor_index(dec)     4 (hex)     4 sensor_identifier(hex)       0
     sensor_index(dec)     5 (hex)     5 sensor_identifier(hex)       0
     sensor_index(dec)     6 (hex)     6 sensor_identifier(hex)       0
     sensor_index(dec)     7 (hex)     7 sensor_identifier(hex)       0
     sensor_index(dec)     8 (hex)     8 sensor_identifier(hex)       0
     sensor_index(dec)     9 (hex)     9 sensor_identifier(hex)       0

    2020-10-15 17:02:34.743 INFO  [9889879] [G4OKTest::init@103] [ setSensorData num_sensor 672 Geometry LIVE TRANSLATED
     sensor_index(dec)     0 (hex)     0 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     1 (hex)     1 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     2 (hex)     2 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     3 (hex)     3 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     4 (hex)     4 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     5 (hex)     5 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     6 (hex)     6 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     7 (hex)     7 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     8 (hex)     8 sensor_identifier(hex)       0 standin(hex)       0
     sensor_index(dec)     9 (hex)     9 sensor_identifier(hex)       0 standin(hex)       0


Issue 5 : FIXED remnant AssimpRap dependency
----------------------------------------------

* curious this build issue did not show up until now 


Issue 5 : 3 test fails
-------------------------

Linux::

    FAILS:  3   / 437   :  Fri Oct 16 03:33:28 2020   
      41 /56  Test #41 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     5.17   
      12 /19  Test #12 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.17   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      8.09   
    [blyth@localhost opticksgeo]$ 

macOS::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         20.86  

    FAILS:  4   / 434   :  Thu Oct 15 20:44:10 2020   
      41 /56  Test #41 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     3.89   
      12 /19  Test #12 : ExtG4Test.X4PhysicalVolume2Test               Child aborted***Exception:     0.12   
      1  /1   Test #1  : G4OKTest.G4OKTest                             ***Exception: SegFault         7.31   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      5.88   
    epsilon:opticks blyth$ 


GGeoIdentityTest 
    FIXED : mm0 is no longer "global" : thats the model change

X4PhysicalVolume2Test 
    FIXED : node collection is deferred so must now use GGeo::getRootVolume

G4OKTest
    FIXED : trivial NULL pv 



One remaining fail

Linux::

    SLOW: tests taking longer that 15 seconds

    FAILS:  1   / 434   :  Fri Oct 16 04:40:54 2020   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      8.13   
    [blyth@localhost opticks]$ 

macOS::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         20.76  

    FAILS:  1   / 434   :  Thu Oct 15 21:44:15 2020   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      5.48   




Issue 6 : IntegrationTests.tboolean.box test geometry SPack assert due to boundary -1 
----------------------------------------------------------------------------------------- 

::

    cd ~/opticks/integration
    om-
    om-test

::

    epsilon:tests blyth$ LV=box tboolean.sh --generateoverride 10000 -D

    2020-10-15 21:29:12.510 INFO  [10234957] [Opticks::loadOriginCacheMeta@1886] ExtractCacheMetaGDMLPath /usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml
    2020-10-15 21:29:12.510 INFO  [10234957] [Opticks::loadOriginCacheMeta@1914] (pass) GEOCACHE_CODE_VERSION 6
    2020-10-15 21:29:12.510 INFO  [10234957] [OpticksHub::loadGeometry@284] [ /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1
    2020-10-15 21:29:12.709 INFO  [10234957] [GNodeLib::GNodeLib@102] loaded
    2020-10-15 21:29:16.119 INFO  [10234957] [NMeta::dump@199] GGeo::loadCacheMeta.lv2sd
    2020-10-15 21:29:16.119 INFO  [10234957] [NMeta::dump@199] GGeo::loadCacheMeta.lv2mt
    2020-10-15 21:29:16.447 INFO  [10234957] [OpticksHub::setupTestGeometry@347] --test modifying geometry
    Assertion failed: ((b & 0xffff0000) == 0), function Encode22, file /Users/blyth/opticks/sysrap/SPack.cc, line 58.
    ...
       frame #3: 0x00007fff589ec1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010b04722e libSysRap.dylib`SPack::Encode22(a=0, b=4294967295) at SPack.cc:58
        frame #5: 0x0000000109dea8d9 libOpticksCore.dylib`OpticksShape::Encode(meshIndex=0, boundaryIndex=4294967295) at OpticksShape.cc:12
        frame #6: 0x000000010984d6b5 libGGeo.dylib`GVolume::getShapeIdentity(this=0x0000000118b457e0) const at GVolume.cc:286
        frame #7: 0x000000010984d651 libGGeo.dylib`GVolume::getIdentity(this=0x0000000118b457e0) const at GVolume.cc:261
        frame #8: 0x000000010989e339 libGGeo.dylib`GNodeLib::addVolume(this=0x0000000119757aa0, volume=0x0000000118b457e0) at GNodeLib.cc:309
        frame #9: 0x0000000109860a4d libGGeo.dylib`GGeoTest::importCSG(this=0x0000000119750b30) at GGeoTest.cc:416
        frame #10: 0x0000000109860128 libGGeo.dylib`GGeoTest::initCreateCSG(this=0x0000000119750b30) at GGeoTest.cc:281
        frame #11: 0x000000010985fa16 libGGeo.dylib`GGeoTest::init(this=0x0000000119750b30) at GGeoTest.cc:175
        frame #12: 0x000000010985f1e7 libGGeo.dylib`GGeoTest::GGeoTest(this=0x0000000119750b30, ok=0x000000010eb5f810, basis=0x000000010ed272c0) at GGeoTest.cc:160
        frame #13: 0x000000010985fc75 libGGeo.dylib`GGeoTest::GGeoTest(this=0x0000000119750b30, ok=0x000000010eb5f810, basis=0x000000010ed272c0) at GGeoTest.cc:154
        frame #14: 0x00000001095d511f libOpticksGeo.dylib`OpticksHub::setupTestGeometry(this=0x000000010ed19ea0) at OpticksHub.cc:353
        frame #15: 0x00000001095d49f4 libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x000000010ed19ea0) at OpticksHub.cc:301
        frame #16: 0x00000001095d336b libOpticksGeo.dylib`OpticksHub::init(this=0x000000010ed19ea0) at OpticksHub.cc:250
        frame #17: 0x00000001095d2fa5 libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ed19ea0, ok=0x000000010eb5f810) at OpticksHub.cc:217
        frame #18: 0x00000001095d35cd libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010ed19ea0, ok=0x000000010eb5f810) at OpticksHub.cc:216
        frame #19: 0x00000001000e071a libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde18) at OKG4Mgr.cc:100
        frame #20: 0x00000001000e0b13 libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde18) at OKG4Mgr.cc:111
        frame #21: 0x0000000100013cb3 OKG4Test`main(argc=32, argv=0x00007ffeefbfde18) at OKG4Test.cc:27
        frame #22: 0x00007fff58978015 libdyld.dylib`start + 1
        frame #23: 0x00007fff58978015 libdyld.dylib`start + 1
    (lldb) 






Add some LEVEL debug to GGeoTest::

    epsilon:tests blyth$ LV=box GGeoTest=INFO tboolean.sh --generateoverride 10000 -D


Ordering problem boundaries only set after importCSG ?::

    251 GMergedMesh* GGeoTest::initCreateCSG()
    252 {
    ...
    281     GVolume* top = importCSG();
    282 
    283     assignBoundaries();
    284 



FIXED : Chicken/egg problem
----------------------------

GGeoTest::assignBoundaries needs the volumes to be in the 
nodelib already but cannot add to nodelib because doing so 
calls GVolume::getIdentity which fails when boundaries are not 
defined yet. 

Fix via: 

* mock up parent child relation between the volumes and 
  use a recursive traverse for assignBoundaries ... 
  to be closer to the normal geometry path ?

Subsequently 

* :doc:`tboolean-box-analysis-fail`

::

    690 void GGeoTest::assignBoundaries()
    691 {
    692     plog::Severity level = m_dbggeotest ? info : debug ;
    693 
    694     LOG(level) << "[" ;
    695 
    696     unsigned numTree = m_csglist->getNumTrees() ;
    697     unsigned numVolume = m_nodelib->getNumVolumes();
    698     assert( numVolume == numTree );
    699 
    700     m_bndlib->closeConstituents();
    701 
    702     for(unsigned i=0 ; i < numVolume ; i++)
    703     {
    704         GVolume* volume = m_nodelib->getVolumeNonConst(i) ;
    705         const NCSG* csg = volume->getMesh()->getCSG();
    706 
    707         const char* spec = csg->getBoundary();
    708         assert(spec);
    709         unsigned boundary = m_bndlib->addBoundary(spec, false);
    710 
    711         volume->setBoundary(boundary); // creates arrays, duplicating boundary to all tris
    712     }
    713 
    714     // see notes/issues/material-names-wrong-python-side.rst
    715     LOG(level) << "Save mlib/slib names "
    716               << " numVolume : " << numVolume
    717               << " csgpath : " << m_csgpath
    718               ;
    719 
    720     if( numVolume > 0 )
    721     {
    722         m_mlib->saveNames(m_csgpath);
    723         m_slib->saveNames(m_csgpath);
    724     }
    725 
    726     LOG(level) << "]" ;
    727 }


