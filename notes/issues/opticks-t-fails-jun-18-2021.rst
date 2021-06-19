opticks-t-fails-jun-18-2021
=============================

::


    SLOW: tests taking longer that 15 seconds
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     53.09  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     65.94  


    FAILS:  5   / 484   :  Sat Jun 19 05:11:31 2021   
      2  /45  Test #2  : OpticksCoreTest.IndexerTest                   ***Exception: SegFault         0.11   

          null boundary data, deluxe data  in saved event 
     

      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     53.09  


      45 /46  Test #45 : CFG4Test.WaterTest                            Child aborted***Exception:     0.26   
            
          FIXED : needs initData

      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     65.94  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.88   
    O[blyth@localhost opticks]$ 



::

    2021-06-19 05:36:18.556 INFO  [63689] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 5 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CGenstepCollector.cc:214: const CGenstep& CGenstepCollector::getGenstep(unsigned int) const: Assertion `gs_idx < m_gs.size()' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007fffe872a387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe872ba78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe87231a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe8723252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b3f911 in CGenstepCollector::getGenstep (this=0x1c28a1f0, gs_idx=0) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:214
    #5  0x00007ffff7b376e9 in CCtx::setTrackOptical (this=0x1b303960, mtrack=0x210abb70) at /home/blyth/opticks/cfg4/CCtx.cc:405
    #6  0x00007ffff7b375e3 in CCtx::setTrack (this=0x1b303960, track=0x210abb70) at /home/blyth/opticks/cfg4/CCtx.cc:376
    #7  0x00007ffff7b3d66a in CManager::PreUserTrackingAction (this=0x1b390680, track=0x210abb70) at /home/blyth/opticks/cfg4/CManager.cc:296
    #8  0x00007ffff7b3625c in CTrackingAction::PreUserTrackingAction (this=0x1b3edc70, track=0x210abb70) at /home/blyth/opticks/cfg4/CTrackingAction.cc:74
    #9  0x00007ffff493908e in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #10 0x00007ffff4b70b53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #11 0x00007ffff4e0db27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #12 0x00007ffff4e06bd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #13 0x00007ffff4e0699e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #14 0x00007ffff7b3ad85 in CG4::propagate (this=0x83e4b30) at /home/blyth/opticks/cfg4/CG4.cc:393
    #15 0x000000000040427e in main (argc=1, argv=0x7fffffffaaa8) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:68
    (gdb) f 4
    #4  0x00007ffff7b3f911 in CGenstepCollector::getGenstep (this=0x1c28a1f0, gs_idx=0) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:214
    214	    assert( gs_idx < m_gs.size() ); 
    (gdb) p m_gs
    $1 = std::vector of length 0, capacity 0
    (gdb) p gs_idx
    $2 = 0
    (gdb) 

