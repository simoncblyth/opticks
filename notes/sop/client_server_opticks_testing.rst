client_server_opticks_testing
================================


TODO
------

* WIP: iteration on NP_CURL.h usage from client

  * ~/o/sysrap/tests/SClientSimulator_test.sh standalone test
  * ~/o/sysrap/tests/SClientSimulatorTest.sh integrated test

  * what about reset : same requirements as current JS runing ?


* build and test JUNOSW against the OPTICKS_CONFIG:Client build "lo_client" (NOT:WITH_CUDA but WITH_CURL, subset of packages + partial packages)

  * should be almost no change on JUNOSW side, maintain same interface


JUNOSW + OpticksClient + OpticksServer
----------------------------------------


junoSD_PMT_v2_Opticks::EndOfEvent_Simulate
    high level use of G4CXOpticks::simulate so should need no change other the Client switch

    * HMM: THE INSTALL SHOULD KNOW IT IS A Client build : SO NO NEED FOR MANUAL CONTROL





server-client testing in brief
---------------------------------

*  server : FastAPI python using nanobind to communicate from python to C++ CSGOptiX instance and back

Build and start the server::

    lo  ## (NOT lo_client) - full Opticks env and newer libcurl than system
    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh


* client 0 : C++ libcurl based using NP_CURL.h NP.hh that repeatedly uploads gensteps loaded from file
* client 1 : curl commandline test client, again uploads gensteps loaded from file

Build and invoke both client tests::

    lo  ## OR "lo_client" - hookup newer curl-config than system one
    ~/np/tests/np_curl_test/np_curl_test.sh



client opticks build
---------------------

* see ~/o/notes/issues/generalizing-build-install-dirs-with-OPTICKS_CONFIG.rst


DONE : shakedown client build
-------------------------------

::

     lo_client
     echo $OPTICKS_PREFIX
     ls -alst $OPTICKS_PREFIX
     rm -rf $OPTICKS_PREFIX    ## make sure are removing the Client prefix

     opticks-full


DONE : libcurl version control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    -- Build files have been written to: /data1/blyth/local/opticks_Client/build/g4cx
    === om-make-one : g4cx            /home/blyth/opticks/g4cx                                     /data1/blyth/local/opticks_Client/build/g4cx
    [  5%] Building CXX object CMakeFiles/G4CX.dir/G4CX_LOG.cc.o
    [ 11%] Building CXX object CMakeFiles/G4CX.dir/G4CXOpticks.cc.o
    In file included from /data1/blyth/local/opticks_Client/include/SysRap/SClientSimulator.h:13,
                     from /home/blyth/opticks/g4cx/G4CXOpticks.cc:35:
    /data1/blyth/local/opticks_Client/include/SysRap/NP_CURL.h: In member function ‘void NP_CURL::perform()’:
    /data1/blyth/local/opticks_Client/include/SysRap/NP_CURL.h:316:43: error: ‘CURLH_HEADER’ was not declared in this scope; did you mean ‘CURLOPT_HEADER’?
      316 |         h = curl_easy_nextheader(session, CURLH_HEADER, -1, p );
          |                                           ^~~~~~~~~~~~
          |                                           CURLOPT_HEADER
    compilation terminated due to -fmax-errors=1.
    make[2]: *** [CMakeFiles/G4CX.dir/build.make:90: CMakeFiles/G4CX.dir/G4CXOpticks.cc.o] Error 1
    make[1]: *** [CMakeFiles/Makefile2:868: CMakeFiles/G4CX.dir/all] Error 2
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === om-all om-cleaninstall : ERROR bdir /data1/blyth/local/opticks_Client/build/g4cx : non-zero rc 2
    === om-one-or-all cleaninstall : non-zero rc 2
    === opticks-full : ERR from opticks-full-make
    (ok) A[blyth@localhost opticks]$

Improve that error message::

    === om-make-one : g4cx            /home/blyth/opticks/g4cx                                     /data1/blyth/local/opticks_Client/build/g4cx
    [  5%] Building CXX object CMakeFiles/G4CX.dir/G4CX_LOG.cc.o
    [ 11%] Building CXX object CMakeFiles/G4CX.dir/G4CXOpticks.cc.o
    In file included from /data1/blyth/local/opticks_Client/include/SysRap/SClientSimulator.h:13,
                     from /home/blyth/opticks/g4cx/G4CXOpticks.cc:35:
    /data1/blyth/local/opticks_Client/include/SysRap/NP_CURL.h:53:2: error: #error "NP_CURL.h libcurl version too old! NP_CURL requires 8.12.1 or higher. Check CMAKE_PREFIX_PATH."
       53 | #error "NP_CURL.h libcurl version too old! NP_CURL requires 8.12.1 or higher. Check CMAKE_PREFIX_PATH."
          |  ^~~~~
    compilation terminated due to -fmax-errors=1.
    make[2]: *** [CMakeFiles/G4CX.dir/build.make:90: CMakeFiles/G4CX.dir/G4CXOpticks.cc.o] Error 1
    make[1]: *** [CMakeFiles/Makefile2:868: CMakeFiles/G4CX.dir/all] Error 2
    make: *** [Makefile:146: all] Error 2
    === om-one-or-all cleaninstall : non-zero rc 2


WIP : realistic client that collects gensteps from Geant4 and uses NP_CURL.h to upload them and download hits
----------------------------------------------------------------------------------------------------------------

Tasks:

* DONE : shakedown the OPTICKS_CONFIG:Client build - a reduced dependency Opticks build : skipping CUDA, OptiX, CSGOptiX

* hit post-processing for localization (summary "muon" hits do not need this)
* how to organize ? OJ interface can stay the same - just need some switch ? SEventConfig::ModeClient OPTICKS_MODE_CLIENT ?


DONE : CMake optional BUILD_WITH_CURL
---------------------------------------

* see ~/np/tests/np_curl_test/np_curl_cmake_test.sh



gx G4CXOpticks WITH_CURL
--------------------------

::

    409 /**
    410 G4CXOpticks::CreateSimulator
    411 ----------------------------
    412
    413 **/
    414
    415
    416 SSimulator* G4CXOpticks::CreateSimulator(CSGFoundry* fd)  // static
    417 {
    418     int64_t mode_client = SEventConfig::ModeClient();
    419     SSimulator* cx = nullptr ;
    420     if( mode_client == 0 )
    421     {
    422 #ifdef WITH_CUDA
    423         cx = CSGOptiX::Create(fd);
    424 #endif
    425     }
    426     else if (mode_client == 1 )
    427     {
    428 #ifdef WITH_CURL
    429          stree* tr = fd->getTree();
    430          cx = new SClientSimulator(tr);
    431 #else
    432          LOG(fatal) << "ModeClient requires compilation WITH_CURL of at least 8.12" ;
    433 #endif
    434     }
    435     assert(cx);
    436     return cx ;
    437 }
    438



SEvt::setHit ? hit/hitmerged/hitlite/hitlitemerged
---------------------------------------------------

::

    4949 const NP* SEvt::getHit() const {       return topfold->get(    SEventConfig::HitCompOneName()) ; }
    4950 size_t    SEvt::getNumHit() const {    return topfold->get_num(SEventConfig::HitCompOneName()) ; }


::

    1441 /**
    1442 SEventConfig::HitCompOne
    1443 -------------------------
    1444
    1445 Canonical usage from::
    1446
    1447     SEvt::getHit
    1448     SEvt::getNumHit
    1449
    1450
    1451 +-----------------------------------------------+--------------------------+---------------------+------------------+
    1452 |   Mode                                        |   HitCompOne             |   HitCompOneName    |  Note            |
    1453 +===============================================+==========================+=====================+==================+
    1454 |    OPTICKS_MODE_LITE=0 OPTICKS_MODE_MERGE=0   |  SCOMP_HIT               |   hit               |                  |
    1455 +-----------------------------------------------+--------------------------+---------------------+------------------+
    1456 |    OPTICKS_MODE_LITE=0 OPTICKS_MODE_MERGE=1   |  SCOMP_HITMERGED         |   hitmerged         |                  |
    1457 +-----------------------------------------------+--------------------------+---------------------+------------------+
    1458 |    OPTICKS_MODE_LITE=1 OPTICKS_MODE_MERGE=0   |  SCOMP_HITLITE           |   hitlite           |                  |
    1459 +-----------------------------------------------+--------------------------+---------------------+------------------+
    1460 |    OPTICKS_MODE_LITE=1 OPTICKS_MODE_MERGE=1   |  SCOMP_HITLITEMERGED     |   hitlitemerged     |                  |
    1461 +-----------------------------------------------+--------------------------+---------------------+------------------+
    1462
    1463 **/
    1464
    1465 unsigned SEventConfig::HitCompOne() // static
    1466 {






SClientSimulator::simulate
----------------------------

::

     69 /**
     70 SClientSimulator::simulate
     71 ---------------------------
     72
     73 TODO: implement this using NP_CURL.h
     74 get gensteps from SEvt, then populate SEvt hits
     75
     76 **/
     77
     78
     79 inline double SClientSimulator::simulate(int eventID, bool reset )
     80 {
     81     assert(eventID > -1);
     82     assert(reset == false);
     83     return 0. ;
     84 }




