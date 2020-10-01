GGeoTest_GMergedMesh_mergeVolumeFaces_assert_sensor_indices
==============================================================


Rerunning integration tests
-----------------------------

::

    cd ~/opticks/integration
    ./tests.sh  ## does om-test  same as "omt" shortcut 


Rerun just the failing one with debugger::

    LV=box tboolean.sh --generateoverride 10000 -D


Issues : Problems with creation and GPU conversion of test geometry 
-----------------------------------------------------------------------

1. FIXED: sensor_indices assert
2. numITransforms == 1 assert


OGeo::convert GMesh::makeFaceRepeatedIdentityBuffer numITransforms == 1 
----------------------------------------------------------------------------

::

    2020-10-01 16:35:41.756 INFO  [898712] [OScene::init@119] [
    2020-10-01 16:35:41.768 INFO  [898712] [OGeo::init@237] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2020-10-01 16:35:41.768 INFO  [898712] [GGeoLib::dump@369] OGeo::convert GGeoLib ANALYTIC  numMergedMesh 1 ptr 0x118e6f750
    mm index   0 geocode   T                  numVolumes          2 numFaces        4408 numITransforms           0 numITransforms*numVolumes           0 GParts Y GPts Y
     num_total_volumes 2 num_instanced_volumes 0 num_global_volumes 2 num_total_faces 4408 num_total_faces_woi 0 (woi:without instancing) 
       0 pts Y  GPts.NumPt     2 lvIdx ( 0 1)
    2020-10-01 16:35:41.768 INFO  [898712] [OGeo::convert@263] [ nmm 1
    2020-10-01 16:35:41.783 INFO  [898712] [*GMesh::makeFaceRepeatedIdentityBuffer@2411]  mmidx 0 numITransforms 0 numVolumes 2 numFaces (sum of faces in numVolumes)4408 numFacesCheck 4408
    Assertion failed: (numITransforms == 1 && "GMesh::makeFaceRepeatedIdentityBuffer only relevant to the non-instanced mm0 "), function makeFaceRepeatedIdentityBuffer, file /Users/blyth/opticks/ggeo/GMesh.cc, line 2420.
    ...
        frame #4: 0x000000010a8c22d4 libGGeo.dylib`GMesh::makeFaceRepeatedIdentityBuffer(this=0x000000010fd4a2c0) at GMesh.cc:2420
        frame #5: 0x000000010a8c1cfb libGGeo.dylib`GMesh::getFaceRepeatedIdentityBuffer(this=0x000000010fd4a2c0) at GMesh.cc:2265
        frame #6: 0x000000010a8c1a66 libGGeo.dylib`GMesh::getAppropriateRepeatedIdentityBuffer(this=0x000000010fd4a2c0) at GMesh.cc:2241
        frame #7: 0x000000010053fd17 libOptiXRap.dylib`OGeo::makeTriangulatedGeometry(this=0x0000000136d48fc0, mm=0x000000010fd4a2c0) at OGeo.cc:938
        frame #8: 0x000000010053e4d3 libOptiXRap.dylib`OGeo::makeOGeometry(this=0x0000000136d48fc0, mergedmesh=0x000000010fd4a2c0) at OGeo.cc:613
        frame #9: 0x000000010053ccf5 libOptiXRap.dylib`OGeo::makeGlobalGeometryGroup(this=0x0000000136d48fc0, mm=0x000000010fd4a2c0) at OGeo.cc:323
        frame #10: 0x000000010053bdb9 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x0000000136d48fc0, i=0) at OGeo.cc:303
        frame #11: 0x000000010053b6fd libOptiXRap.dylib`OGeo::convert(this=0x0000000136d48fc0) at OGeo.cc:269
        frame #12: 0x0000000100531649 libOptiXRap.dylib`OScene::init(this=0x000000011d6eb1d0) at OScene.cc:169
        frame #13: 0x0000000100530a01 libOptiXRap.dylib`OScene::OScene(this=0x000000011d6eb1d0, hub=0x000000010fb1dae0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:91
        frame #14: 0x0000000100531c1d libOptiXRap.dylib`OScene::OScene(this=0x000000011d6eb1d0, hub=0x000000010fb1dae0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:90
        frame #15: 0x0000000100441c16 libOKOP.dylib`OpEngine::OpEngine(this=0x000000011d6eb110, hub=0x000000010fb1dae0) at OpEngine.cc:75
        frame #16: 0x000000010044230d libOKOP.dylib`OpEngine::OpEngine(this=0x000000011d6eb110, hub=0x000000010fb1dae0) at OpEngine.cc:83
        frame #17: 0x0000000100108faf libOK.dylib`OKPropagator::OKPropagator(this=0x000000011d6e6c80, hub=0x000000010fb1dae0, idx=0x000000011d5695a0, viz=0x0000000114ac5170) at OKPropagator.cc:68
        frame #18: 0x000000010010915d libOK.dylib`OKPropagator::OKPropagator(this=0x000000011d6e6c80, hub=0x000000010fb1dae0, idx=0x000000011d5695a0, viz=0x0000000114ac5170) at OKPropagator.cc:72
        frame #19: 0x00000001000e199f libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde18) at OKG4Mgr.cc:110
        frame #20: 0x00000001000e1b13 libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde18) at OKG4Mgr.cc:111
        frame #21: 0x0000000100014c73 OKG4Test`main(argc=32, argv=0x00007ffeefbfde18) at OKG4Test.cc:27
    (lldb) 





GGeoTest::initCreateCSG GMergedMesh::mergeVolumeFaces sensor_indices=0x0000000000000000
------------------------------------------------------------------------------------------

First problem sensor_indices assert, fixed by GVolume::setSensorIndex::


    2020-10-01 15:58:21.353 INFO  [758068] [NMeta::dump@199] GGeo::loadCacheMeta.lv2mt
    2020-10-01 15:58:21.359 INFO  [758068] [OpticksHub::loadGeometry@559] --test modifying geometry
    2020-10-01 15:58:21.359 INFO  [758068] [GNodeLib::GNodeLib@72] created
    Assertion failed: (sensor_indices), function mergeVolumeFaces, file /Users/blyth/opticks/ggeo/GMergedMesh.cc, line 876.
        frame #4: 0x000000010a90b233 libGGeo.dylib`GMergedMesh::mergeVolumeFaces(this=0x00000001146ddc40, nface=12, faces=0x00000001146d88c0, node_indices=0x00000001146d9620, boundary_indices=0x00000001146dda20, sensor_indices=0x0000000000000000) at GMergedMesh.cc:876
        frame #5: 0x000000010a908960 libGGeo.dylib`GMergedMesh::mergeVolume(this=0x00000001146ddc40, volume=0x00000001146d91a0, selected=true, verbosity=1) at GMergedMesh.cc:606
        frame #6: 0x000000010a90a0bb libGGeo.dylib`GMergedMesh::traverse_r(this=0x00000001146ddc40, node=0x00000001146d91a0, depth=0, pass=1, verbosity=1) at GMergedMesh.cc:398
        frame #7: 0x000000010a909ac9 libGGeo.dylib`GMergedMesh::Create(ridx=0, base=0x0000000000000000, root=0x00000001146d91a0, verbosity=1, globalinstance=false) at GMergedMesh.cc:318
        frame #8: 0x000000010a8e433e libGGeo.dylib`GGeoTest::initCreateCSG(this=0x000000011a10cf70) at GGeoTest.cc:279
        frame #9: 0x000000010a8e3c04 libGGeo.dylib`GGeoTest::init(this=0x000000011a10cf70) at GGeoTest.cc:164
        frame #10: 0x000000010a8e3551 libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000011a10cf70, ok=0x000000010fd5fa60, basis=0x000000010fc004c0) at GGeoTest.cc:155
        frame #11: 0x000000010a8e3e25 libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000011a10cf70, ok=0x000000010fd5fa60, basis=0x000000010fc004c0) at GGeoTest.cc:149
        frame #12: 0x00000001095db94f libOpticksGeo.dylib`OpticksHub::createTestGeometry(this=0x000000010fb1dae0, basis=0x000000010fc004c0) at OpticksHub.cc:613
        frame #13: 0x00000001095da1cb libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x000000010fb1dae0) at OpticksHub.cc:565
        frame #14: 0x00000001095d8bbe libOpticksGeo.dylib`OpticksHub::init(this=0x000000010fb1dae0) at OpticksHub.cc:253
        frame #15: 0x00000001095d87fb libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010fb1dae0, ok=0x000000010fd5fa60) at OpticksHub.cc:217
        frame #16: 0x00000001095d8ded libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010fb1dae0, ok=0x000000010fd5fa60) at OpticksHub.cc:216
        frame #17: 0x00000001000e171a libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde10) at OKG4Mgr.cc:100
        frame #18: 0x00000001000e1b13 libOKG4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007ffeefbfdd40, argc=32, argv=0x00007ffeefbfde10) at OKG4Mgr.cc:111
        frame #19: 0x0000000100014c73 OKG4Test`main(argc=32, argv=0x00007ffeefbfde10) at OKG4Test.cc:27
    (lldb) 

    (lldb) list 870
       870 	**/
       871 	
       872 	void GMergedMesh::mergeVolumeFaces( unsigned nface, guint3* faces, unsigned* node_indices, unsigned* boundary_indices, unsigned* sensor_indices )
       873 	{
       874 	    assert(node_indices);
       875 	    assert(boundary_indices);
       876 	    assert(sensor_indices);
       877 	
       878 	    for(unsigned i=0 ; i < nface ; ++i )
       879 	    {
    (lldb) 

    (lldb) f 5
    frame #5: 0x000000010a908960 libGGeo.dylib`GMergedMesh::mergeVolume(this=0x00000001146ddc40, volume=0x00000001146d91a0, selected=true, verbosity=1) at GMergedMesh.cc:606
       603 	        unsigned* boundary_indices = volume->getBoundaryIndices();
       604 	        unsigned* sensor_indices   = volume->getSensorIndices();
       605 	
    -> 606 	        mergeVolumeFaces( num_face, faces, node_indices, boundary_indices, sensor_indices  ); // m_faces, m_nodes, m_boundaries, m_sensors
       607 	   
       608 	#ifdef GPARTS_HOT 
       609 	        assert(0) ; // THIS OLD WAY WAS TERRIBLY WASTEFUL : INSTEAD MOVED TO DEFERRED GParts CONCAT USING GPt WHICH COLLECTS THE ARGS FOR GParts  
    (lldb) 




Where do the sensor indices normally get set ?
-------------------------------------------------

::

    131 unsigned int* GNode::getSensorIndices() const
    132 {
    133     return m_sensor_indices ;
    134 }

    325 void GNode::setSensorIndices(unsigned int index)
    326 {
    327     // unsigned int* array of the node index repeated nface times
    328     unsigned int nface = m_mesh->getNumFaces();
    329     unsigned int* indices = new unsigned int[nface] ;
    330     while(nface--) indices[nface] = index ;
    331     m_sensor_indices = indices ;
    332 }

    epsilon:tests blyth$ opticks-f setSensorIndices
    ./ggeo/GNode.cc:void GNode::setSensorIndices(unsigned int index)
    ./ggeo/GVolume.cc:    setSensorIndices( m_sensor_index );   // GNode::setSensorIndices duplicate to all faces of m_mesh triangulated geometry
    ./ggeo/GVolume.cc:    setSensorIndices( NSensor::RefIndex(sensor) );
    ./ggeo/GNode.hh:setSensorIndices
    ./ggeo/GNode.hh:      void setSensorIndices(unsigned int sensor_index);
    ./ggeo/GNode.hh:      void setSensorIndices(unsigned int* sensor_indices);
    epsilon:opticks blyth$ 


    261 void GVolume::setSensorIndex(int sensor_index)
    262 {
    263     m_sensor_index = sensor_index ;
    264     setSensorIndices( m_sensor_index );   // GNode::setSensorIndices duplicate to all faces of m_mesh triangulated geometry
    265 }

    epsilon:opticks blyth$ opticks-f setSensorIndex
    ./extg4/X4PhysicalVolume.cc:    volume->setSensorIndex(sensorIndex); 
    ./ggeo/GVolume.cc:void GVolume::setSensorIndex(int sensor_index)
    ./ggeo/GVolume.hh:      void     setSensorIndex(int sensor_index) ;
    epsilon:opticks blyth$ 


::

    1200 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1201 {
    ...
    1213     unsigned boundary = addBoundary( pv, pv_p );
    1214     std::string boundaryName = m_blib->shortname(boundary);
    ...
    1359     int sensorIndex = m_blib->isSensorBoundary(boundary) ? m_ggeo->addSensorVolume(volume) : -1 ;
    1360     if(sensorIndex > -1) m_blib->countSensorBoundary(boundary);
    1361 
    1362     /*
    1363     if(sensorIndex > -1)
    1364     {
    1365         LOG(info)
    1366             << " copyNumber " << std::setw(8) << copyNumber
    1367             << " sensorIndex " << std::setw(8) << sensorIndex
    1368             << " boundary " << std::setw(4) << boundary 
    1369             << " boundaryName " << boundaryName
    1370             ;
    1371     }
    1372     */
    1373 
    1374     volume->setSensorIndex(sensorIndex);



    0529 bool GBndLib::isSensorBoundary(unsigned boundary) const
     530 {
     531     const guint4& bnd = m_bnd[boundary];
     532     bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]);
     533     bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]);
     534     bool is_sensor = osur_sensor || isur_sensor ;
     535     return is_sensor ;
     536 }

    epsilon:extg4 blyth$ opticks-f isSensorIndex
    ./ggeo/GBndLib.cc:    bool osur_sensor = m_slib->isSensorIndex(bnd[OSUR]); 
    ./ggeo/GBndLib.cc:    bool isur_sensor = m_slib->isSensorIndex(bnd[ISUR]); 
    ./ggeo/GPropertyLib.cc:bool GPropertyLib::isSensorIndex(unsigned index) const 
    ./ggeo/GSurfaceLib.cc:            assert( isSensorIndex(i) == true ) ; 
    ./ggeo/GPropertyLib.hh:        bool isSensorIndex(unsigned index) const ; 
    epsilon:opticks blyth$ 

    898 // m_sensor_indices is a transient (non-persisted) vector of material/surface indices 
    899 bool GPropertyLib::isSensorIndex(unsigned index) const
    900 {
    901     typedef std::vector<unsigned>::const_iterator UI ;
    902     UI b = m_sensor_indices.begin();
    903     UI e = m_sensor_indices.end();
    904     UI i = std::find(b, e, index);
    905     return i != e ;
    906 }

    908 /**
    909 GPropertyLib::addSensorIndex
    910 ------------------------------
    911 
    912 Canonically invoked from GSurfaceLib::collectSensorIndices
    913 
    914 **/
    915 void GPropertyLib::addSensorIndex(unsigned index)
    916 {
    917     m_sensor_indices.push_back(index);
    918 }
    919 unsigned GPropertyLib::getNumSensorIndices() const
    920 {
    921     return m_sensor_indices.size();
    922 }
    923 unsigned GPropertyLib::getSensorIndex(unsigned i) const
    924 {
    925     return m_sensor_indices[i] ;
    926 }
    927 void GPropertyLib::dumpSensorIndices(const char* msg) const
    928 {
    929     unsigned ni = getNumSensorIndices() ;
    930     std::stringstream ss ;
    931     ss << " NumSensorIndices " << ni << " ( " ;
    932     for(unsigned i=0 ; i < ni ; i++) ss << getSensorIndex(i) << " " ;
    933     ss << " ) " ;
    934     std::string desc = ss.str();
    935     LOG(info) << msg << " " << desc ;
    936 }

    0723 void GSurfaceLib::collectSensorIndices()
     724 {
     725     unsigned ni = getNumSurfaces();
     726     for(unsigned i=0 ; i < ni ; i++)
     727     {
     728         GPropertyMap<float>* surf = m_surfaces[i] ;
     729         bool is_sensor = surf->isSensor() ;
     730         if(is_sensor)
     731         {
     732             addSensorIndex(i);
     733             assert( isSensorIndex(i) == true ) ;
     734         }
     735     }
     736 }
     737 

    0288 template <class T>
     289 bool GPropertyMap<T>::isSensor()
     290 {
     291 #ifdef OLD_SENSOR
     292     return m_sensor ;
     293 #else
     294     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     295 #endif
     296 }






