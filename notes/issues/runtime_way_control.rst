runtime_way_control
=================================


Objective 
------------

Add runtime control inside the inconvenient on their own compiletime switches.

WITH_WAY_BUFFER
   way point recording 
   --way Opticks::isWayEnabled



avoid quadupling sources with oxrap/cu/preprocessor.py which generates as specified by flags 
---------------------------------------------------------------------------------------------

oxrap/CMakeLists.txt::

    set(flags_AW +WITH_ANGULAR,+WITH_WAY_BUFFER)
    set(flags_Aw +WITH_ANGULAR,-WITH_WAY_BUFFER)
    set(flags_aW -WITH_ANGULAR,+WITH_WAY_BUFFER)
    set(flags_aw -WITH_ANGULAR,-WITH_WAY_BUFFER)


    add_custom_command(
        OUTPUT  generate_${flags_AW}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_AW}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_AW}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_Aw}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_Aw}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_Aw}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_aW}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_aW}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_aW}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_aw}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_aw}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_aw}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )


checking are using the generated cu and ptx
----------------------------------------------

::

    epsilon:optixrap blyth$ l /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_*
    -rw-r--r--  1 blyth  staff  480674 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  480348 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  482497 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  478825 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx

::

    epsilon:opticks blyth$ OContext=INFO OConfig=INFO OKTest 


    2021-02-16 22:51:17.450 INFO  [11247997] [OpSeeder::seedComputeSeedsFromInteropGensteps@82] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2021-02-16 22:51:17.480 INFO  [11247997] [OConfig::createProgram@114]  cu_name generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu progname generate m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:51:17.651 INFO  [11247997] [OConfig::createProgram@114]  cu_name generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:51:17.659 INFO  [11247997] [OConfig::createProgram@114]  cu_name pinhole_camera.cu progname pinhole_camera m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2021-02-16 22:51:17.777 INFO  [11247997] [OConfig::createProgram@114]  cu_name pinhole_camera.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2021-02-16 22:51:17.781 INFO  [11247997] [OConfig::createProgram@114]  cu_name constantbg.cu progname miss m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_constantbg.cu.ptx



    epsilon:opticks blyth$ OContext=INFO OConfig=INFO OKTest --way --angular --compute

    2021-02-16 22:53:41.453 ERROR [11250092] [SensorLib::close@362]  SKIP as m_sensor_num zero 
    2021-02-16 22:53:41.479 FATAL [11250092] [*OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_sensor_data
    2021-02-16 22:53:41.505 FATAL [11250092] [*OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_texid
    2021-02-16 22:53:41.541 INFO  [11250092] [OConfig::createProgram@114]  cu_name generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu progname generate m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:53:41.729 INFO  [11250092] [OConfig::createProgram@114]  cu_name generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:53:41.737 INFO  [11250092] [OContext::launch@810]  entry 0 width 0 height 0   printLaunchIndex ( -1 -1 -1) -
    2021-02-16 22:53:41.740 INFO  [11250092] [OContext::launch@823] VALIDATE time: 0.003286



Some Fails as the default does not inclyde --way
----------------------------------------------------

Both fails from the same assert::

 

::


    FAILS:  2   / 448   :  Wed Feb 17 07:23:06 2021   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     10.49  
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     13.81  
    [blyth@localhost opticks]$ 

    Test project /home/blyth/local/opticks/build/g4ok
        Start 1: G4OKTest.G4OKTest
    1/2 Test #1: G4OKTest.G4OKTest ................Child aborted***Exception:  13.81 sec

    2021-02-17 07:22:45.068 INFO  [212887] [OEvent::downloadHits@471]  nhit 53 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    TBuf::download SKIP  numItems_tbuf 0
    CBufSpec.Summary (empty tbuf?) : dev_ptr (nil) size 0 num_bytes 0 hexdump 0 
    create_empty_npy
    2021-02-17 07:22:45.069 INFO  [212887] [OEvent::downloadHiys@506]  nhiy 0 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-02-17 07:22:45.069 FATAL [212887] [OEvent::download@542]  nhit 53 nhiy 0
    G4OKTest: /home/blyth/opticks/optixrap/OEvent.cc:548: unsigned int OEvent::download(): Assertion nhit == nhiy failed.


::

    527 unsigned OEvent::download()
    528 {
    529     LOG(LEVEL) << "[" ;
    530 
    531     if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);
    532 
    533     unsigned nhit = downloadHits();
    534     LOG(LEVEL) << " nhit " << nhit ;
    535 
    536 #ifdef WITH_WAY_BUFFER
    537     unsigned nhiy = downloadHiys();
    538     LOG(LEVEL)
    539         << " nhiy " << nhiy ;
    540     if( nhit != nhiy )
    541     {
    542         LOG(fatal)
    543             << " nhit " << nhit
    544             << " nhiy " << nhiy
    545             ;
    546     }
    547 
    548     assert( nhit == nhiy );
    549 #endif
    550 
    551     LOG(LEVEL) << "]" ;
    552     return nhit ;
    553 }


* TODO: check all WITH_WAY_BUFFER WITH_ANGULAR and add runtime checks inside them 


Migrate from compile time WITH_WAY_BUFFER to --way option and Opticks::isWayEnabled and WAY_ENABLED preprocessor.py flag 
----------------------------------------------------------------------------------------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/commits/eaebae4a59ee859b85c49ccb2a2d9e4c38e14f3e


::

    epsilon:opticks blyth$ opticks-f WITH_WAY_BUFFER
    ./g4ok/G4Opticks.cc:#ifdef WITH_WAY_BUFFER
    ./g4ok/G4Opticks.cc:#ifdef WITH_WAY_BUFFER

    handling the hiys

    0480 void G4Opticks::reset()
     481 {
     482     resetCollectors();
     483 
     484     m_hits->reset();   // the cloned hits (and hiys) are owned by G4Opticks, so they must be reset here  
     485 #ifdef WITH_WAY_BUFFER
     486     m_hiys->reset();
     487 #endif
     488 
     489 }

    1068 #ifdef WITH_WAY_BUFFER
    1069         m_hiys = event->getHiyData()->clone() ;
    1070         m_num_hiys = m_hits->getNumItems() ;
    1071         LOG(fatal) << " WAY_BUFFER num_hiys " << m_num_hiys ;
    1072         m_hits->setAux(m_hiys);   // associate the extra hiy selected from way buffer with hits array 
    1073 #else
    1074         LOG(fatal) << " no-WAY_BUFFER " ;
    1075 #endif



    ./g4ok/G4OpticksHit.hh:when WITH_WAY_BUFFER from optickscore/OpticksSwitches.h 
    ./ggeo/GPho.cc:The way array is only available when optickscore/OpticksSwitches.h:WITH_WAY_BUFFER is defined. 

    comments only 

    ./optickscore/OpticksSwitches.h:#define WITH_WAY_BUFFER 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_WAY_BUFFER
    ./optickscore/OpticksSwitches.h:    ss << "WITH_WAY_BUFFER " ;   

    define and present  

    ./optickscore/OpticksCfg.cc:       ("way",     "enable way/hiy point recording at runtime, requires the WITH_WAY_BUFFER compile time switch to be enabled") ;

    runtime control

    ./optixrap/CMakeLists.txt:set(flags_AW +WITH_ANGULAR,+WITH_WAY_BUFFER)
    ./optixrap/CMakeLists.txt:set(flags_Aw +WITH_ANGULAR,-WITH_WAY_BUFFER)
    ./optixrap/CMakeLists.txt:set(flags_aW -WITH_ANGULAR,+WITH_WAY_BUFFER)
    ./optixrap/CMakeLists.txt:set(flags_aw -WITH_ANGULAR,-WITH_WAY_BUFFER)

    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER
    ./optixrap/cu/generate.cu:#ifdef WITH_WAY_BUFFER

    ./optixrap/cu/state.h:#ifdef WITH_WAY_BUFFER
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
    state.h in included into generate.cu making this one problematic because generate.cu 
    gets preprocessed into multiple with various flag settings for runtime switching 
    of OptiX raygen program
    
    * so need to bodily include state.h into generate.cu 

    ./optixrap/cu/preprocessor.py:of flag macros, eg WITH_ANGULAR WITH_WAY_BUFFER 
    ./optixrap/cu/preprocessor.py:    parser.add_argument( "-f", "--flags", help="Comma delimited control flags eg +WITH_WAY_BUFFER,-WITH_ANGULAR " )
    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,+WITH_WAY_BUFFER
    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,-WITH_WAY_BUFFER
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,+WITH_WAY_BUFFER
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,-WITH_WAY_BUFFER

    ./optixrap/OContext.cc:        << w << "WITH_WAY_BUFFER"

    forming the filename 



    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:    LOG(LEVEL) << "WITH_WAY_BUFFER way " << way->getShapeString() ; 
    ./optixrap/OEvent.cc:    LOG(LEVEL) << "not WITH_WAY_BUFFER " ; 
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.cc:#ifdef WITH_WAY_BUFFER




    ./optixrap/OEvent.hh:#if defined(WITH_DEBUG_BUFFER) && defined(WITH_WAY_BUFFER)
    ./optixrap/OEvent.hh:#elif defined(WITH_WAY_BUFFER)
    ./optixrap/OEvent.hh:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.hh:#ifdef WITH_WAY_BUFFER
    ./optixrap/OEvent.hh:#ifdef WITH_WAY_BUFFER

    111 class OXRAP_API OEvent
    112 {
    113     public:
    114         static const plog::Severity LEVEL ;
    115     public:
    116         enum {
    117             GENSTEP  = 0x1 << 1,
    118             PHOTON   = 0x1 << 2,
    119             RECORD   = 0x1 << 3,
    120             SEQUENCE = 0x1 << 4,
    121             SEED     = 0x1 << 5,
    122             SOURCE   = 0x1 << 6,
    123             DEBUG    = 0x1 << 7,
    124             WAY      = 0x1 << 8,
    125 #if defined(WITH_DEBUG_BUFFER) && defined(WITH_WAY_BUFFER)
    126             DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE | DEBUG | WAY
    127 #elif defined(WITH_WAY_BUFFER)
    128             DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE | WAY
    129 #elif defined(WITH_DEBUG_BUFFER)
    130             DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE | DEBUG
    131 #else
    132             DOWNLOAD_DEFAULT  = PHOTON | RECORD | SEQUENCE
    133 #endif
    134             };

    * TODO: DOWNLOAD_DEFAULT setup at runtime

    151 #ifdef WITH_WAY_BUFFER
    152     public:
    153         unsigned downloadHiys();
    154     private:
    155         unsigned downloadHiysCompute(OpticksEvent* evt);
    156         unsigned downloadHiysInterop(OpticksEvent* evt);
    157 #endif

    196 #ifdef WITH_DEBUG_BUFFER
    197         optix::Buffer   m_debug_buffer ;
    198 #endif
    199 #ifdef WITH_WAY_BUFFER
    200         optix::Buffer   m_way_buffer ;
    201 #endif

    * just leave empty buffer ?





opticks-t

::

    FAILS:  1   / 448   :  Wed Feb 17 19:01:06 2021   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     13.08  
    [blyth@localhost opticks]$ 


    2021-02-17 19:00:40.755 INFO  [372726] [OEvent::downloadHits@485]  nhit 53 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-02-17 19:00:40.755 INFO  [372726] [OpticksEvent::save@1809] /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1
    2021-02-17 19:00:40.763 FATAL [372726] [G4Opticks::propagateOpticalPhotons@1081]  NOT-m_way_enabled 
    2021-02-17 19:00:40.764 INFO  [372726] [G4OKTest::getNumGenstepPhotons@324]  eventID 0 num 5000
    2021-02-17 19:00:40.764 ERROR [372726] [G4OKTest::propagate@349]  eventID 0 num_genstep_photons 5000 num_hit 53
    2021-02-17 19:00:40.764 INFO  [372726] [G4OKTest::checkHits@378]  eventID 0 num_gensteps 0 num_photons 0 num_hit 53
    G4OKTest: /home/blyth/opticks/g4ok/G4Opticks.cc:1203: void G4Opticks::getHit(unsigned int, G4OpticksHit*, G4OpticksHitExtra*) const: Assertion `m_hits_wrapper->hasWay()' failed.

        Start 2: G4OKTest.G4OpticksHitTest
    2/2 Test #2: G4OKTest.G4OpticksHitTest ........   Passed    4.76 sec


    [blyth@localhost optickscore]$ OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--way" G4OKTest 



