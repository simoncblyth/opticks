OSensorLib_texid_missing
==========================




::

    OKTest 

    2020-12-01 12:42:35.540 NONE  [2889187] [OpticksViz::uploadEvent@406] [ (0)
    2020-12-01 12:42:35.570 NONE  [2889187] [OpticksViz::uploadEvent@413] ] (0)
    2020-12-01 12:42:35.723 INFO  [2889187] [OpSeeder::seedComputeSeedsFromInteropGensteps@83] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable OSensorLib_texid from _Z8generatev_cp6" not found in scope)
    Abort trap: 6
    epsilon:opticks blyth$ 


::

    FAILS:  3   / 452   :  Tue Dec  1 20:43:10 2020   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.72   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     14.05  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      13.11  
    [blyth@localhost opticks]$ 


Note that G4OKTest does not fail.

Including the OSensorLib.hh header into OptiX kernels requires OSensorLib::convert to be called
to populate the context, that happens via call to OScene::uploadSensorLib


::

     OSensorLib=INFO OKTest 


::

    epsilon:optixrap blyth$ opticks-f uploadSensorLib
    ./okop/OpEngine.cc:void OpEngine::uploadSensorLib(const SensorLib* sensorlib)
    ./okop/OpEngine.cc:    m_scene->uploadSensorLib(sensorlib); 
    ./okop/OpPropagator.hh:       void uploadSensorLib(const SensorLib* sensorlib); 
    ./okop/OpMgr.cc:void OpMgr::uploadSensorLib(const SensorLib* sensorlib)
    ./okop/OpMgr.cc:    m_propagator->uploadSensorLib(sensorlib);  
    ./okop/OpMgr.hh:       void uploadSensorLib(const SensorLib* sensorlib); 
    ./okop/OpEngine.hh:       void uploadSensorLib(const SensorLib* sensorlib);
    ./okop/OpPropagator.cc:void OpPropagator::uploadSensorLib(const SensorLib* sensorlib)
    ./okop/OpPropagator.cc:    m_engine->uploadSensorLib(sensorlib); 

    ./g4ok/G4Opticks.cc:G4Opticks::uploadSensorLib
    ./g4ok/G4Opticks.cc:void G4Opticks::uploadSensorLib() 
    ./g4ok/G4Opticks.cc:    assert( m_opmgr && "must setGeometry and set sensor info before uploadSensorLib" ); 
    ./g4ok/G4Opticks.cc:    m_opmgr->uploadSensorLib(m_sensorlib); 
    ./g4ok/tests/G4OKTest.cc:    m_g4ok->uploadSensorLib(); 
    ./g4ok/G4Opticks.hh:        void uploadSensorLib() ;

    ./optixrap/OSensorLib.hh:It is instancianted and converted by OScene::uploadSensorLib
    ./optixrap/OScene.hh:       void uploadSensorLib(const SensorLib* sensorlib); 
    ./optixrap/OScene.cc:void OScene::uploadSensorLib(const SensorLib* sensorlib)
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



G4OKTest
----------

::

    136 void G4OKTest::init()
    137 {
    138     initGeometry();
    139     initSensorData();
    140     initSensorAngularEfficiency();
    141     if(m_debug) saveSensorLib();
    142     m_g4ok->uploadSensorLib();
    143 



