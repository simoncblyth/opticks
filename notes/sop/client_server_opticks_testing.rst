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

    * DONE : INSTALL USES THE MACROS TO AVOID NEED FOR CLIENT SWITCH




TODO: build JUNOSW + OpticksClient on workstation - local.sh function "ljbb_client" ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prime change is prefix::

    #export JUNO_OPTICKS_PREFIX=/data1/blyth/local/opticks_Debug
    export JUNO_OPTICKS_PREFIX=/data1/blyth/local/opticks_Client



TODO : how to make OpticksClient release onto cvmfs ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* decide on naming and layout of OpticksClient tarball and cvmfs folders


TODO : how to make JUNOSW+OpticksClient release onto cvmfs ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





DONE: geometry consistency check via tree digest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    stree::get_tree_digest()


* need to pass root hash from client with gensteps and return root hash from server with hits

  * discrepancy should kill the client, not the server



LOOKS OK : will localization work in client ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    437 void junoSD_PMT_v2_Opticks::EndOfEvent_CollectFullHits_premerged(int eventID, const SEvt* sev, const sphoton* hit, size_t num_hit )
    438 {
    439     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_premerged_HEAD");
    440     junoHit_PMT_Collection* hitCollection = m_jpmt->getHitCollection() ;
    441     assert( hitCollection );
    442
    443     for(size_t i=0 ; i < num_hit ; i++)
    444     {
    445         const sphoton& p = hit[i];
    446         sphoton l = p ;
    447         sev->localize_photon_inplace(l);
    448
    449         junoHit_PMT* hit = new junoHit_PMT();
    450         PopulateFullHit(hit, l, p );
    451         hitCollection->insert(hit);
    452     }
    453     std::string anno = SProf::Annotation("hit",num_hit);
    454     SProf::Add("junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_premerged_TAIL", anno.c_str());
    455 }

    5222 void SEvt::localize_photon_inplace( sphoton& p ) const
    5223 {
    5224     assert(tree);
    5225     tree->localize_photon_inplace(p);
    5226 }


    7929 inline void stree::localize_photon_inplace( sphoton& p ) const
    7930 {
    7931     unsigned iindex   = p.iindex() ;
    7932     assert( iindex != 0xffffffffu );
    7933     const glm::tmat4x4<double>* tr = get_iinst(iindex) ;
    7934     assert( tr );
    7935
    7936     bool normalize = true ;
    7937     p.transform( *tr, normalize );   // inplace transforms l (pos, mom, pol) into local frame
    7938
    7939 #ifdef NDEBUG
    7940 #else
    7941     unsigned sensor_identifier = p.pmtid() ;
    7942
    7943     glm::tvec4<int64_t> col3 = {} ;
    7944     strid::Decode( *tr, col3 );
    7945
    7946     sphit ht = {};
    7947     ht.iindex            = col3[0] ;
    7948     ht.sensor_identifier = col3[2] ;
    7949     ht.sensor_index      = col3[3] ;
    7950
    7951     assert( ht.iindex == iindex );
    7952     assert( ht.sensor_identifier == sensor_identifier );
    7953 #endif
    7954
    7955 }









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




