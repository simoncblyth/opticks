opticks_t_44_of_486_after_switch_to_double_props
==================================================

Old G4 tests need adjustments for CManager
----------------------------------------------

::

    FAILS:  3   / 486   :  Mon Jun 28 17:27:52 2021   
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     40.34  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     52.23  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      5.48   


Some CCtx::step_limit requires CCtx::initEvent to have been called
-----------------------------------------------------------------------




::

    102	unsigned CCtx::step_limit() const 
    103	{
    104	    assert( _ok_event_init ); 
    105	    return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
    106	}


    205 /**
    206 CCtx::initEvent
    207 --------------------
    208 
    209 Collect the parameters of the OpticksEvent which 
    210 dictate what needs to be collected.
    211 
    212 **/
    213 
    214 void CCtx::initEvent(const OpticksEvent* evt)
    215 {
    216     _ok_event_init = true ;
    217     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    218     _steps_per_photon = evt->getMaxRec() ;   // number of points to be recorded into record buffer   
    219     _record_max = evt->getNumPhotons();      // from the genstep summation, hmm with dynamic running this will start as zero 
    220 
    221     _bounce_max = evt->getBounceMax();       // maximum bounce allowed before truncation will often be 1 less than _steps_per_photon but need not be 
    222     unsigned bounce_max_2 = evt->getMaxBounce();
    223     assert( _bounce_max == bounce_max_2 ) ; // TODO: eliminate or rename one of those
    224 
    225     const char* typ = evt->getTyp();
    226 
    227     LOG(LEVEL)
    228         << " _record_max (numPhotons from genstep summation) " << _record_max
    229         << " photons_per_g4event " << _photons_per_g4event
    230         << " _steps_per_photon (maxrec) " << _steps_per_photon
    231         << " _bounce_max " << _bounce_max
    232         << " typ " << typ
    233         ;
    234 
    235 }





::

    2021-06-28 17:35:39.648 ERROR [28420] [main@63]  setting gensteps 0x731ddb0 numPhotons 20000
    2021-06-28 17:35:39.649 INFO  [28420] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-06-28 17:35:39.649 INFO  [28420] [main@68]  cgs T  idx   0 pho20000 off      0
    2021-06-28 17:35:39.653 INFO  [28420] [CG4::propagate@396]  calling BeamOn numG4Evt 1
    2021-06-28 17:36:11.397 INFO  [28420] [CScint::Check@16]  pmanager 0x7539510 proc 0
    2021-06-28 17:36:11.397 INFO  [28420] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-06-28 17:36:11.397 INFO  [28420] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007fffe8738387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe8739a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe87311a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe8731252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b3580d in CCtx::step_limit (this=0x7516b40) at /home/blyth/opticks/cfg4/CCtx.cc:104
    #5  0x00007ffff7acaf9c in CRec::add (this=0x75170d0, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRec.cc:286
    #6  0x00007ffff7b0ff58 in CRecorder::Record (this=0x190a1210, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRecorder.cc:344
    #7  0x00007ffff7b3cef4 in CManager::setStep (this=0x1901b980, step=0x74a4970) at /home/blyth/opticks/cfg4/CManager.cc:502
    #8  0x00007ffff7b3cb48 in CManager::UserSteppingAction (this=0x1901b980, step=0x74a4970) at /home/blyth/opticks/cfg4/CManager.cc:429
    #9  0x00007ffff7b34a42 in CSteppingAction::UserSteppingAction (this=0x19bbc510, step=0x74a4970) at /home/blyth/opticks/cfg4/CSteppingAction.cc:41
    #10 0x00007ffff492a9a2 in G4SteppingManager::Stepping() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #11 0x00007ffff49360fd in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #12 0x00007ffff4b6db53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #13 0x00007ffff4e0ab27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #14 0x00007ffff4e03bd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #15 0x00007ffff4e0399e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #16 0x00007ffff7b39fc9 in CG4::propagate (this=0x731e350) at /home/blyth/opticks/cfg4/CG4.cc:399
    #17 0x0000000000404526 in main (argc=1, argv=0x7fffffff9188) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:76
    (gdb) 




CGenstep bookkeeping
-----------------------

::

    epsilon:opticks blyth$ git commit -m "add new CGenstep bookkeeping to old tests CG4Test OKG4Test "
    [master 36976ea1f] add new CGenstep bookkeeping to old tests CG4Test OKG4Test



CGenstepCollector::addGenstep not being called with CG4Test ?
--------------------------------------------------------------


CG4Test::

     49     CG4* g4 = new CG4(&hub) ;
     50     LOG(warning) << " post CG4 " ;
     51 
     52     g4->interactive();
     53 
     54     LOG(warning) << "  post CG4::interactive"  ;
     55 
     56     if(ok.isFabricatedGensteps())  // eg TORCH running
     57     {
     58         NPY<float>* gs = gen->getInputGensteps() ;
     59         LOG(error) << " setting gensteps " << gs ;
     60         char ctrl = '=' ;
     61         ok.createEvent(gs, ctrl);
     62     }
     63     else
     64     {
     65         LOG(error) << " not setting gensteps " ;
     66     }
     67 
     68     g4->propagate();
     69 
     70     LOG(info) << "  CG4 propagate DONE "  ;
     71 
     72     ok.postpropagate();


::


    296 CGenstep CGenstepCollector::addGenstep(unsigned numPhotons, char gentype)
    297 {
    298     unsigned genstep_index = getNumGensteps();
    299     unsigned photon_offset = getNumPhotons();
    300 
    301     CGenstep gs(genstep_index, numPhotons, photon_offset, gentype) ;
    302 
    303     LOG(LEVEL) << " gs.desc " << gs.desc() ;
    304 
    305     m_gs.push_back(gs);
    306     m_gs_photons.push_back(numPhotons);
    307     m_gs_offset.push_back(photon_offset);
    308     m_gs_type.push_back(gentype);
    309 
    310     m_photon_count += numPhotons ;
    311 
    312     CManager* mgr = CManager::Get();
    313     if(mgr && (gentype == 'C' || gentype == 'S'))
    314     {
    315         mgr->BeginOfGenstep(genstep_index, gentype, numPhotons, photon_offset);
    316     }
    317 
    318     return gs  ;
    319 }




CG4Test : with old simple torchstep failing for lack of the new genstep bookkeeping
----------------------------------------------------------------------------------------

::

    2021-06-28 16:48:06.840 INFO  [401595] [OpticksHub::loadGeometry@315] ]
    2021-06-28 16:48:06.841 INFO  [401595] [Opticks::makeSimpleTorchStep@4163] [ts.setFrameTransform
    ...

    2021-06-28 16:48:12.145 INFO  [401595] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-06-28 16:48:12.150 INFO  [401595] [CG4::propagate@390]  calling BeamOn numG4Evt 1
    2021-06-28 16:48:43.626 INFO  [401595] [CScint::Check@16]  pmanager 0x7539880 proc 0
    2021-06-28 16:48:43.627 INFO  [401595] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-06-28 16:48:43.627 INFO  [401595] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CGenstepCollector.cc:214: const CGenstep& CGenstepCollector::getGenstep(unsigned int) const: Assertion `gs_idx < m_gs.size()' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007fffe8738387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe8739a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe87311a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe8731252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b3ead1 in CGenstepCollector::getGenstep (this=0x9309590, gs_idx=0) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:214
    #5  0x00007ffff7b36867 in CCtx::setTrackOptical (this=0x1901bdf0, mtrack=0x1ba0bb60) at /home/blyth/opticks/cfg4/CCtx.cc:424
    #6  0x00007ffff7b366ff in CCtx::setTrack (this=0x1901bdf0, track=0x1ba0bb60) at /home/blyth/opticks/cfg4/CCtx.cc:379
    #7  0x00007ffff7b3c81a in CManager::PreUserTrackingAction (this=0x1906e570, track=0x1ba0bb60) at /home/blyth/opticks/cfg4/CManager.cc:299
    #8  0x00007ffff7b3536a in CTrackingAction::PreUserTrackingAction (this=0x1914cf10, track=0x1ba0bb60) at /home/blyth/opticks/cfg4/CTrackingAction.cc:74
    #9  0x00007ffff493608e in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #10 0x00007ffff4b6db53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #11 0x00007ffff4e0ab27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #12 0x00007ffff4e03bd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #13 0x00007ffff4e0399e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #14 0x00007ffff7b39f33 in CG4::propagate (this=0x731e4f0) at /home/blyth/opticks/cfg4/CG4.cc:393
    #15 0x000000000040427e in main (argc=1, argv=0x7fffffff92d8) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:68
    (gdb) 

    (gdb) f 4
    #4  0x00007ffff7b3ead1 in CGenstepCollector::getGenstep (this=0x9309590, gs_idx=0) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:214
    214	    assert( gs_idx < m_gs.size() ); 
    (gdb) p gs_idx
    $1 = 0
    (gdb) p m_gs.size()
    $2 = 0
    (gdb) 




After recreate geocache are down to 3 fails
----------------------------------------------


::

    SLOW: tests taking longer that 15 seconds
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     40.12  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     51.55  


    FAILS:  3   / 486   :  Mon Jun 28 16:44:22 2021   
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     40.12  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     51.55  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.47   
    O[blyth@localhost 1]$ 



okc/Opticks.cc bump Opticks::GEOCACHE_CODE_VERSION to 10 to force recreation of geocache
--------------------------------------------------------------------------------------------

::

     404 geocache-jun28-gdmlpath(){ echo $(opticks-prefix)/origin_CGDMLKludge_jun28.gdml ; }
     405 geocache-jun28(){
     406     local msg="=== $FUNCNAME :"
     407     local path=$(geocache-jun28-gdmlpath)
     408     # get skips from current tds3
     409     local skipsolidname="mask_PMT_20inch_vetosMask_virtual,NNVTMCPPMT_body_solid,HamamatsuR12860_body_solid_1_9,PMT_20inch_veto_body_solid_1_2"
     410     GTree=INFO OpticksDbg=INFO GInstancer=INFO geocache-create- --gdmlpath $path -D --noviz  --skipsolidname $skipsolidname $*  
     411 }   


* After geocache-jun28 and changing the default OPTICKS_KEY GScintillatorLibTest is passing on eps.
* On Gold update Opticks and run tds3gun to use the current default settings for creating a new geocache. 



Lots of errors from failed array loading due to expecting double
-------------------------------------------------------------------

::

    FAILS:  44  / 486   :  Mon Jun 28 07:39:16 2021   
      13 /58  Test #13 : GGeoTest.GScintillatorLibTest                 Child aborted***Exception:     0.09   
      16 /58  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.08   
      17 /58  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.08   
      31 /58  Test #31 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.41   
      35 /58  Test #35 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.08   
      40 /58  Test #40 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.07   
      41 /58  Test #41 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.07   
      42 /58  Test #42 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     0.08   
      43 /58  Test #43 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     0.07   
      45 /58  Test #45 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.06   
      52 /58  Test #52 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.06   
      54 /58  Test #54 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.08   
      57 /58  Test #57 : GGeoTest.GPhoTest                             Child aborted***Exception:     0.07   
      58 /58  Test #58 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     0.08   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.08   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.08   
      3  /3   Test #3  : OpticksGeoTest.OpticksHubGGeoTest             Child aborted***Exception:     0.35   
      3  /35  Test #3  : OptiXRapTest.OScintillatorLibTest             Child aborted***Exception:     0.19   
      11 /35  Test #11 : OptiXRapTest.textureTest                      Child aborted***Exception:     0.18   
      12 /35  Test #12 : OptiXRapTest.boundaryTest                     Child aborted***Exception:     0.20   
      13 /35  Test #13 : OptiXRapTest.reemissionTest                   Child aborted***Exception:     0.20   
      15 /35  Test #15 : OptiXRapTest.boundaryLookupTest               Child aborted***Exception:     0.23   
      19 /35  Test #19 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.19   
      24 /35  Test #24 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.17   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.19   
      1  /6   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.19   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.19   
      5  /6   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.19   
      6  /6   Test #6  : OKOPTest.OpFlightPathTest                     Child aborted***Exception:     0.19   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.20   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.21   
      1  /46  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.75   
      2  /46  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.25   
      3  /46  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.26   
      5  /46  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.24   
      7  /46  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.22   
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     0.24   
      28 /46  Test #28 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.40   
      30 /46  Test #30 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.23   
      38 /46  Test #38 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.23   
      39 /46  Test #39 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.24   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.28   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     0.25   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      3.25   
    O[blyth@localhost opticks]$ 

