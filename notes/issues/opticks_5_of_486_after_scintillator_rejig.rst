opticks_5_of_486_after_scintillator_rejig TWO FIXED, THREE REMAIN
=====================================================================


* geocache created via tds3gun after bump to Opticks::GEOCACHE_CODE_VERSION = 13




::


    SLOW: tests taking longer that 15 seconds
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     42.00  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     52.51  


    FAILS:  5   / 486   :  Mon Jul  5 18:24:45 2021   
      48 /58  Test #48 : GGeoTest.GMaterialTest                        Child aborted***Exception:     0.07   
      9  /30  Test #9  : ExtG4Test.X4MaterialTest                      Child aborted***Exception:     0.15   
                    FIXED

      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     42.00  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     52.51  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      4.90   
    O[blyth@localhost opticks]$ 




GMaterialTest : twas deliberate assert to find callers of low level method, now made private, FIXED
------------------------------------------------------------------------------------------------------


::

    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/GMaterialTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    GPropertyMap<T>::pdigest unexpected ifr/ito 0/0 
    GMaterial::Summary material 0 d41d8cd98f00b204e9800998ecf8427e test
    GMaterialTest: /home/blyth/opticks/ggeo/GPropertyMap.cc:653: void GPropertyMap<T>::addPropertyStandardized(const char*, T*, T*, unsigned int, const char*) [with T = double]: Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #3  0x00007ffff48ff252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7abd0fe in GPropertyMap<double>::addPropertyStandardized (this=0x617470, pname=0x40521d "pname", values=0x7fffffffaed0, domain=0x7fffffffae90, length=7, prefix=0x0)
        at /home/blyth/opticks/ggeo/GPropertyMap.cc:653
    #5  0x00000000004028c0 in test_addProperty () at /home/blyth/opticks/ggeo/tests/GMaterialTest.cc:40
    #6  0x0000000000402a73 in main (argc=1, argv=0x7fffffffb068) at /home/blyth/opticks/ggeo/tests/GMaterialTest.cc:52
    (gdb) 


X4MaterialTest : old call of X4MaterialPropertiesTable::Convert with a bool for the mode, FIXED
--------------------------------------------------------------------------------------------------

::


    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/X4MaterialTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2021-07-05 18:41:41.807 FATAL [177681] [X4MaterialPropertiesTable::AddProperties@116]  mode must be one of G/S/A 
    X4MaterialTest: /home/blyth/opticks/extg4/X4MaterialPropertiesTable.cc:117: static void X4MaterialPropertiesTable::AddProperties(GPropertyMap<double>*, const G4MaterialPropertiesTable*, char): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #3  0x00007fffeda4c252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b7e6cf in X4MaterialPropertiesTable::AddProperties (pmap=0x6c7130, mpt=0x6c2ad0, mode=1 '\001') at /home/blyth/opticks/extg4/X4MaterialPropertiesTable.cc:117
    #5  0x00007ffff7b7e2a7 in X4MaterialPropertiesTable::init (this=0x7fffffffac40) at /home/blyth/opticks/extg4/X4MaterialPropertiesTable.cc:54
    #6  0x00007ffff7b7e273 in X4MaterialPropertiesTable::X4MaterialPropertiesTable (this=0x7fffffffac40, pmap=0x6c7130, mpt=0x6c2ad0, mode=1 '\001')
        at /home/blyth/opticks/extg4/X4MaterialPropertiesTable.cc:49
    #7  0x00007ffff7b7e202 in X4MaterialPropertiesTable::Convert (pmap=0x6c7130, mpt=0x6c2ad0, mode=1 '\001') at /home/blyth/opticks/extg4/X4MaterialPropertiesTable.cc:40
    #8  0x00007ffff7b7aa40 in X4Material::init (this=0x7fffffffaec0) at /home/blyth/opticks/extg4/X4Material.cc:137
    #9  0x00007ffff7b7a762 in X4Material::X4Material (this=0x7fffffffaec0, material=0x6c26a0, mode=1 '\001') at /home/blyth/opticks/extg4/X4Material.cc:90
    #10 0x00007ffff7b7a6b8 in X4Material::Convert (material=0x6c26a0, mode=1 '\001') at /home/blyth/opticks/extg4/X4Material.cc:69
    #11 0x0000000000402b57 in main (argc=1, argv=0x7fffffffb058) at /home/blyth/opticks/extg4/tests/X4MaterialTest.cc:40
    (gdb) 



CG4Test + OKG4Test : still needing to be brought into new Genstep bookkeeping approach 
-----------------------------------------------------------------------------------------


::

    2021-07-05 18:49:05.398 WARN  [188272] [main@52]  post CG4 
    2021-07-05 18:49:05.398 WARN  [188272] [main@56]   post CG4::interactive
    2021-07-05 18:49:05.399 ERROR [188272] [main@63]  setting gensteps 0x9b2f460 numPhotons 20000
    2021-07-05 18:49:05.399 INFO  [188272] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-07-05 18:49:05.400 INFO  [188272] [main@68]  cgs T  idx   0 pho20000 off      0
    2021-07-05 18:49:05.403 INFO  [188272] [CG4::propagate@396]  calling BeamOn numG4Evt 1
    2021-07-05 18:49:37.023 INFO  [188272] [CScint::Check@16]  pmanager 0x1b8111f0 proc 0
    2021-07-05 18:49:37.023 INFO  [188272] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-07-05 18:49:37.024 INFO  [188272] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #3  0x00007fffe872f252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b3582f in CCtx::step_limit (this=0x1b848a40) at /home/blyth/opticks/cfg4/CCtx.cc:104
    #5  0x00007ffff7acafac in CRec::add (this=0x1b848cb0, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRec.cc:286
    #6  0x00007ffff7b0ff74 in CRecorder::Record (this=0x9d1ce80, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRecorder.cc:345
    #7  0x00007ffff7b3cf16 in CManager::setStep (this=0x9d976d0, step=0x9cb3280) at /home/blyth/opticks/cfg4/CManager.cc:502
    #8  0x00007ffff7b3cb6a in CManager::UserSteppingAction (this=0x9d976d0, step=0x9cb3280) at /home/blyth/opticks/cfg4/CManager.cc:429
    #9  0x00007ffff7b34a64 in CSteppingAction::UserSteppingAction (this=0x1b8968b0, step=0x9cb3280) at /home/blyth/opticks/cfg4/CSteppingAction.cc:41
    #10 0x00007ffff492a9a2 in G4SteppingManager::Stepping() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #11 0x00007ffff49360fd in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #12 0x00007ffff4b6db53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #13 0x00007ffff4e0ab27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #14 0x00007ffff4e03bd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #15 0x00007ffff4e0399e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #16 0x00007ffff7b39feb in CG4::propagate (this=0x9b2fa00) at /home/blyth/opticks/cfg4/CG4.cc:399
    #17 0x0000000000404526 in main (argc=1, argv=0x7fffffffb068) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:76
    (gdb) 







