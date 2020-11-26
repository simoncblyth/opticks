G4OKTest_CANNOT_createBufferFromGLBO_as_not_uploaded_gensteps
==============================================================


::

    epsilon:opticks blyth$ lldb_ G4OKTest 
    (lldb) target create "G4OKTest"
    Current executable set to 'G4OKTest' (x86_64).
    (lldb) r
    Process 36122 launched: '/usr/local/opticks/lib/G4OKTest' (x86_64)
    2020-11-26 16:01:03.325 INFO  [360374] [G4Opticks::G4Opticks@290] ctor : DISABLE FPE detection : as it breaks OptiX launches

      C4FPEDetection::InvalidOperationDetection_Disable       NOT IMPLEMENTED 
    2020-11-26 16:01:03.327 INFO  [360374] [BOpticksKey::SetKey@77]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-11-26 16:01:03.327 INFO  [360374] [*G4Opticks::InitOpticks@193] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave 
    ...
    2020-11-26 16:01:03.328 INFO  [360374] [Opticks::init@428] INTEROP_MODE hostname epsilon.local
    2020-11-26 16:01:07.207 INFO  [360374] [GGeo::dumpStats@1362] GGeo::Load
     mm  0 : vertices  247718 faces  480972 transforms    4486 itransforms       1 
     mm  1 : vertices       8 faces      12 transforms       1 itransforms    1792 
     mm  2 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  3 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  4 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  5 : vertices    1498 faces    2976 transforms       5 itransforms     672 
       totVertices    249248  totFaces    483996 
      vtotVertices   1289446 vtotFaces   2533452 (virtual: scaling by transforms)
      vfacVertices     5.173 vfacFaces     5.234 (virtual to total ratio)
    2020-11-26 16:01:07.207 INFO  [360374] [G4Opticks::setGeometry@494]  GGeo: LOADED FROM CACHE  num_sensor 12229
    2020-11-26 16:01:07.208 WARN  [360374] [OpticksGen::initFromLegacyGensteps@192] SKIP as isNoInputGensteps OR isEmbedded  
    2020-11-26 16:01:08.006 INFO  [360374] [OContext::CheckDevices@196] 
    Device 0                GeForce GT 750M ordinal 0 Compute Support: 3 0 Total Memory: 2147024896

    2020-11-26 16:01:08.013 INFO  [360374] [CDevice::Dump@244] Visible devices[0:GeForce_GT_750M]
    2020-11-26 16:01:08.013 INFO  [360374] [CDevice::Dump@248] CDevice index 0 ordinal 0 name GeForce GT 750M major 3 minor 0 compute_capability 30 multiProcessorCount 2 totalGlobalMem 2147024896
    2020-11-26 16:01:08.014 INFO  [360374] [CDevice::Dump@244] All devices[0:GeForce_GT_750M]
    2020-11-26 16:01:08.014 INFO  [360374] [CDevice::Dump@248] CDevice index 0 ordinal 0 name GeForce GT 750M major 3 minor 0 compute_capability 30 multiProcessorCount 2 totalGlobalMem 2147024896
    2020-11-26 16:01:08.061 INFO  [360374] [OGeo::init@236] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    ...
    020-11-26 16:01:10.564 DEBUG [360374] [G4OKTest::initSensorData@189] [ setSensorData num_sensor 12229 num_distinct_copynumber 0 Geometry LOADED FROM CACHE
    2020-11-26 16:01:10.566 DEBUG [360374] [G4OKTest::initSensorData@221] ] setSensorData num_sensor 12229
    2020-11-26 16:01:10.567 INFO  [360374] [G4Opticks::saveSensorLib@679]  saving to $TMP/G4OKTest/SensorLib
    2020-11-26 16:01:10.567 INFO  [360374] [SensorLib::save@38] $TMP/G4OKTest/SensorLib
    2020-11-26 16:01:10.568 INFO  [360374] [G4OKTest::saveSensorLib@238] $TMP/G4OKTest/SensorLib
    2020-11-26 16:01:10.568 INFO  [360374] [G4Opticks::uploadSensorLib@693] 
    2020-11-26 16:01:10.568 INFO  [360374] [SensorLib::checkSensorCategories@374] [ SensorLib closed N loaded N sensor_data 12229,4 sensor_num 12229 sensor_angular_efficiency 1,180,360,1 num_category 1
    2020-11-26 16:01:10.570 INFO  [360374] [SensorLib::dumpCategoryCounts@412] SensorLib::checkSensorCategories
     category          0 count      12229
    2020-11-26 16:01:10.570 INFO  [360374] [SensorLib::checkSensorCategories@407] ] SensorLib closed N loaded N sensor_data 12229,4 sensor_num 12229 sensor_angular_efficiency 1,180,360,1 num_category 1
    2020-11-26 16:01:10.570 INFO  [360374] [SensorLib::close@369] SensorLib closed Y loaded N sensor_data 12229,4 sensor_num 12229 sensor_angular_efficiency 1,180,360,1 num_category 1
    2020-11-26 16:01:10.605 INFO  [360374] [OSensorLib::makeSensorAngularEfficiencyTexture@107]  item 0 tex_id 4
    2020-11-26 16:01:10.635 INFO  [360374] [GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName@260]  target_lvname /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140 nidxs.size() 2 nidx 3153
    2020-11-26 16:01:10.635 ERROR [360374] [G4OKTest::collectGensteps@277]  eventID 0 num_genstep_photons 5000
    2020-11-26 16:01:10.635 DEBUG [360374] [G4OKTest::propagate@285] [
    2020-11-26 16:01:10.636 FATAL [360374] [OpPropagator::propagate@71] evtId(0) OK INTEROP PRODUCTION
    2020-11-26 16:01:10.636 FATAL [360374] [OContext::createBuffer@1086] CANNOT createBufferFromGLBO as not uploaded   name             gensteps buffer_id -1
    Assertion failed: (buffer_id > -1), function createBuffer, file /Users/blyth/opticks/optixrap/OContext.cc, line 1091.
    Process 36122 stopped
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff659a2b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff65b6d080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff658fe1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff658c61ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010646c67b libOptiXRap.dylib`optix::Handle<optix::BufferObj> OContext::createBuffer<float>(this=0x00000001181b6240, npy=0x0000000118524f90, name="gensteps") at OContext.cc:1091
        frame #5: 0x0000000106483a4b libOptiXRap.dylib`OEvent::createBuffers(this=0x000000011d2a61d0, evt=0x000000011d2a7e50) at OEvent.cc:145
        frame #6: 0x000000010648532e libOptiXRap.dylib`OEvent::upload(this=0x000000011d2a61d0, evt=0x000000011d2a7e50) at OEvent.cc:345
        frame #7: 0x000000010648518e libOptiXRap.dylib`OEvent::upload(this=0x000000011d2a61d0) at OEvent.cc:334
        frame #8: 0x0000000106391d15 libOKOP.dylib`OpEngine::uploadEvent(this=0x000000011852a430) at OpEngine.cc:171
        frame #9: 0x0000000106394510 libOKOP.dylib`OpPropagator::uploadEvent(this=0x000000011852a3d0) at OpPropagator.cc:91
        frame #10: 0x000000010639427c libOKOP.dylib`OpPropagator::propagate(this=0x000000011852a3d0) at OpPropagator.cc:73
        frame #11: 0x0000000106392fc3 libOKOP.dylib`OpMgr::propagate(this=0x0000000118525a20) at OpMgr.cc:136
        frame #12: 0x00000001000e90df libG4OK.dylib`G4Opticks::propagateOpticalPhotons(this=0x000000010e81deb0, eventID=0) at G4Opticks.cc:914
        frame #13: 0x00000001000129eb G4OKTest`G4OKTest::propagate(this=0x00007ffeefbfe880, eventID=0) at G4OKTest.cc:286
        frame #14: 0x0000000100013bd2 G4OKTest`main(argc=1, argv=0x00007ffeefbfe8e8) at G4OKTest.cc:356
        frame #15: 0x00007fff65852015 libdyld.dylib`start + 1
        frame #16: 0x00007fff65852015 libdyld.dylib`start + 1
    (lldb) 
      [Restored Nov 26, 2020 at 4:05:40 PM]
    Last login: Thu Nov 26 16:05:40 on ttys003



::

     866 int G4Opticks::propagateOpticalPhotons(G4int eventID)
     867 {
     868     LOG(LEVEL) << "[[" ;
     869     assert( m_genstep_collector );
     870     m_gensteps = m_genstep_collector->getGensteps();
     871     m_gensteps->setArrayContentVersion(G4VERSION_NUMBER);
     872     m_gensteps->setArrayContentIndex(eventID);
     873 
     874     unsigned num_gensteps = m_gensteps->getNumItems();
     875     LOG(LEVEL) << " num_gensteps "  << num_gensteps ;
     876     if( num_gensteps == 0 )
     877     {   
     878         LOG(fatal) << "SKIP as no gensteps have been collected " ;
     879         return 0 ;
     880     }
     881 
     882 
     883     unsigned tagoffset = eventID ;  // tags are 1-based : so this will normally be the Geant4 eventID + 1
     884 
     885     if(!m_ok->isProduction()) // --production
     886     {   
     887         const char* gspath = m_ok->getDirectGenstepPath(tagoffset);
     888         LOG(LEVEL) << "[ saving gensteps to " << gspath ;
     889         m_gensteps->save(gspath);  
     890         LOG(LEVEL) << "] saving gensteps to " << gspath ;
     891     }



This is bizarre, as "--compute" mode is active there should be no OpenGL buffers anyhow ?
Perhaps OpticksMode expecting a commandline with "--compute" on it, so detection not working ?

Indeed yes, looks like inconsistent compute vs interop modes causing issue::

    OpticksMode=INFO Opticks=INFO lldb_ G4OKTest  
    ...
    2020-11-26 16:47:25.175 INFO  [67240] [BOpticksKey::SetKey@77]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-11-26 16:47:25.175 INFO  [67240] [*G4Opticks::InitOpticks@193] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave 
    2020-11-26 16:47:25.175 INFO  [67240] [OpticksMode::OpticksMode@108] INTEROP_MODE
    2020-11-26 16:47:25.175 INFO  [67240] [Opticks::envkey@322] 
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::init@427] [
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::init@428] INTEROP_MODE hostname epsilon.local
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::init@437]  mandatory keyed access to geometry, opticksaux 
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::init@457] ]
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2375] Opticks::configure argc 5
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2377]   0 : --compute
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2377]   1 : --embedded
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2377]   2 : --xanalytic
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2377]   3 : --production
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::dumpArgs@2377]   4 : --nosave
    2020-11-26 16:47:25.176 INFO  [67240] [Opticks::initResource@861] [ OpticksResource 

Fixing Opticks::hasArg to use SArgs::hasArg which looks at extraline as well as commandline 
should avoid the problem.

::

     892 /**   
     893 Opticks::hasArg
     894 -----------------
     895       
     896 The old hasArg only looked at the actual argv commandline arguments not the 
     897 combination of commandline and extraline (aka argforced) that SArgs::hasArg checks.
     898 As embedded running such as G4Opticks uses the extraline to configure Opticks
     899 it is vital to check with m_sargs.
     900 
     901 **/
     902 
     903 bool Opticks::hasArg(const char* arg) const
     904 {   
     905     return m_sargs->hasArg(arg);
     906 }     


Confirmed fix::

    epsilon:cfg4 blyth$ OpticksMode=INFO Opticks=INFO lldb_ G4OKTest 
    ...
    2020-11-26 17:09:26.369 INFO  [104087] [*G4Opticks::InitOpticks@193] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave 
    2020-11-26 17:09:26.370 INFO  [104087] [OpticksMode::OpticksMode@108] COMPUTE_MODE compute_requested 
    2020-11-26 17:09:26.370 INFO  [104087] [Opticks::envkey@322] 
    2020-11-26 17:09:26.370 INFO  [104087] [Opticks::init@427] [
    2020-11-26 17:09:26.370 INFO  [104087] [Opticks::init@428] COMPUTE_MODE compute_requested  hostname epsilon.local
    2020-11-26 17:09:26.370 INFO  [104087] [Opticks::init@437]  mandatory keyed access to geometry, opticksaux 
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::init@457] ]
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2385] Opticks::configure argc 5
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2387]   0 : --compute
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2387]   1 : --embedded
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2387]   2 : --xanalytic
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2387]   3 : --production
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::dumpArgs@2387]   4 : --nosave
    2020-11-26 17:09:26.371 INFO  [104087] [Opticks::initResource@861] [ OpticksResource 


