opticks-t-fails-aug-2021-13-of-493
======================================

Aug 25 11:39 13/493 
----------------------

::

    SLOW: tests taking longer that 15 seconds
      31 /31  Test #31 : ExtG4Test.X4SurfaceTest                       Passed                         45.15        REDUCED TEST SIZE
      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93  


    FAILS:  13  / 493   :  Wed Aug 25 18:39:55 2021   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      4.97     FINDING PYTHON WITH NUMPY 

      18 /31  Test #18 : ExtG4Test.X4CSGTest                           ***Exception: SegFault         0.13     FIXED WITH local_tempStr
      20 /31  Test #20 : ExtG4Test.X4GDMLParserTest                    ***Exception: SegFault         0.14   
      21 /31  Test #21 : ExtG4Test.X4GDMLBalanceTest                   ***Exception: SegFault         0.15   
      32 /46  Test #32 : CFG4Test.CTreeJUNOTest                        ***Exception: SegFault         0.22     SAME ISSUE : IT USES GDML SNIPPET WRITING    



      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.21     LACK OF INIT WITH TORCH GENSTEPS
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  67.93       

        2021-08-25 19:40:24.060 INFO  [90759] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
        CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.



      1  /46  Test #1  : CFG4Test.CMaterialLibTest                     Subprocess aborted***Exception:   2.40      SCINTILLATOR REJIG ISSUE
      2  /46  Test #2  : CFG4Test.CMaterialTest                        Subprocess aborted***Exception:   2.38   
      30 /46  Test #30 : CFG4Test.CGROUPVELTest                        Subprocess aborted***Exception:   2.44   
      38 /46  Test #38 : CFG4Test.CCerenkovGeneratorTest               Subprocess aborted***Exception:   2.38   
      39 /46  Test #39 : CFG4Test.CGenstepSourceTest                   Subprocess aborted***Exception:   2.35   




      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.87   
    O[blyth@localhost opticks]$ 



Aug 25 16:16 : Now there are 4/493
-------------------------------------

::


    FAILS:  4   / 493   :  Wed Aug 25 23:15:49 2021   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                ***Failed                      4.96         ## py: No numpy module  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.90         ## py: No module named 'opticks'

      8  /46  Test #8  : CFG4Test.CG4Test                              Subprocess aborted***Exception:  53.31  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Subprocess aborted***Exception:  66.72  
    O[blyth@localhost cfg4]$ 




CG4Test + OKG4Test : need to call the init with torch gensteps   
----------------------------------------------------------------

::

    2021-08-25 23:14:40.956 INFO  [436572] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-08-25 23:14:40.956 INFO  [436572] [OKG4Mgr::propagate_@222]  numPhotons 20000 cgs T  idx   0 pho20000 off      0
    2021-08-25 23:14:40.968 INFO  [436572] [CG4::propagate@396]  calling BeamOn numG4Evt 1
    2021-08-25 23:15:29.375 INFO  [436572] [CScint::Check@16]  pmanager 0xae6a000 proc 0
    2021-08-25 23:15:29.375 INFO  [436572] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-08-25 23:15:29.375 INFO  [436572] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    OKG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.


    2021-08-25 23:28:15.555 INFO  [457070] [CScint::Check@16]  pmanager 0x1d83b7d0 proc 0
    2021-08-25 23:28:15.556 INFO  [457070] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-08-25 23:28:15.556 INFO  [457070] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.

    (gdb) bt
    #3  0x00007fffe8787252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b36add in CCtx::step_limit (this=0xab34680) at /home/blyth/opticks/cfg4/CCtx.cc:104
    #5  0x00007ffff7acc530 in CRec::add (this=0x1d864d60, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRec.cc:286
    #6  0x00007ffff7b1123c in CRecorder::Record (this=0x1d864c60, boundary_status=FresnelRefraction) at /home/blyth/opticks/cfg4/CRecorder.cc:345
    #7  0x00007ffff7b3e1c4 in CManager::setStep (this=0x1d835120, step=0xaac0cc0) at /home/blyth/opticks/cfg4/CManager.cc:502
    #8  0x00007ffff7b3de18 in CManager::UserSteppingAction (this=0x1d835120, step=0xaac0cc0) at /home/blyth/opticks/cfg4/CManager.cc:429
    #9  0x00007ffff7b35d12 in CSteppingAction::UserSteppingAction (this=0xa94ad60, step=0xaac0cc0) at /home/blyth/opticks/cfg4/CSteppingAction.cc:41
    #10 0x00007ffff4936ba2 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #11 0x00007ffff49409cd in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #12 0x00007ffff4b76f61 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #13 0x00007ffff4e0ee87 in G4RunManager::ProcessOneEvent(int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #14 0x00007ffff4e080f3 in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #15 0x00007ffff4e07ebe in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #16 0x00007ffff7b3b299 in CG4::propagate (this=0xa934430) at /home/blyth/opticks/cfg4/CG4.cc:399
    #17 0x0000000000404556 in main (argc=1, argv=0x7fffffff65e8) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:76
    (gdb) 




::

    102 unsigned CCtx::step_limit() const
    103 {
    104     assert( _ok_event_init );
    105     return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
    106 }

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


    238 /**
    239 CManager::initEvent : configure event recording, limits/shapes etc.. 
    240 ------------------------------------------------------------------------
    241 
    242 Invoked from CManager::BeginOfEventAction/CManager::presave
    243 
    244 **/
    245 
    246 void CManager::initEvent(OpticksEvent* evt)
    247 {
    248     LOG(LEVEL) << " m_mode " << m_mode ;
    249     assert( m_mode > 1 );
    250 
    251     m_ctx->initEvent(evt);
    252     m_recorder->initEvent(evt);
    253 
    254     NPY<float>* nopstep = evt->getNopstepData();
    255     if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ;
    256     assert(nopstep);
    257     m_noprec->initEvent(nopstep);
    258 }



Huh CEventAction should have called that::

     45 void CEventAction::BeginOfEventAction(const G4Event* event)
     46 {
     47     m_manager->BeginOfEventAction(event);
     48 }

::

    O[blyth@localhost cfg4]$ export CEventAction=INFO
    O[blyth@localhost cfg4]$ export CManager=INFO
    O[blyth@localhost cfg4]$ gdb CG4Test

    2021-08-25 23:42:59.142 INFO  [22136] [CManager::BeginOfRunAction@110]  m_mode 3
    2021-08-25 23:42:59.142 INFO  [22136] [CScint::Check@16]  pmanager 0x1d83b900 proc 0
    2021-08-25 23:42:59.143 INFO  [22136] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-08-25 23:42:59.143 INFO  [22136] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    2021-08-25 23:42:59.154 INFO  [22136] [CManager::BeginOfEventAction@130]  m_mode 3
    2021-08-25 23:42:59.463 INFO  [22136] [CManager::BeginOfEventAction@142]  not calling presave, creating OpticksEvent 
    CG4Test: /home/blyth/opticks/cfg4/CCtx.cc:104: unsigned int CCtx::step_limit() const: Assertion `_ok_event_init' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe878e387 in raise () from /lib64/libc.so.6


Hmm looks like the problem is lack of "--save" probably from a change of default::

    128 void CManager::BeginOfEventAction(const G4Event* event)
    129 {
    130     LOG(LEVEL) << " m_mode " << m_mode ;
    131     if(m_mode == 0 ) return ;
    132 
    133     m_ctx->setEvent(event);
    134 
    135     if(m_ok->isSave())
    136     {
    137         LOG(LEVEL) << " calling presave to create OpticksEvent " ;
    138         presave();   // creates the OpticksEvent
    139     }
    140     else
    141     {
    142         LOG(LEVEL) << " not calling presave, creating OpticksEvent " ;
    143     }
    144 


Gets further with "--save" but lots of "[CWriter::writeStepPoint@207]  SKIP  unexpected record_id 9999 m_ni 65"::

    O[blyth@localhost cfg4]$ gdb --args CG4Test --save
    ...
    2021-08-25 23:48:06.118 INFO  [29968] [CManager::BeginOfRunAction@110]  m_mode 3
    2021-08-25 23:48:06.119 INFO  [29968] [CScint::Check@16]  pmanager 0x1d83bc30 proc 0
    2021-08-25 23:48:06.119 INFO  [29968] [CScint::Check@21] CProMgr n:[4] (0) name Transportation left -1 (1) name OpAbsorption left -1 (2) name OpRayleigh left -1 (3) name OpBoundary left -1
    2021-08-25 23:48:06.119 INFO  [29968] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 6 : BAD_FLAG
    2021-08-25 23:48:06.130 INFO  [29968] [CManager::BeginOfEventAction@130]  m_mode 3
    2021-08-25 23:48:06.422 INFO  [29968] [CManager::BeginOfEventAction@137]  calling presave to create OpticksEvent 
    2021-08-25 23:48:06.422 INFO  [29968] [CManager::presave@217]  mode 3
    2021-08-25 23:48:06.422 INFO  [29968] [CManager::presave@223]  [--save] creating OpticksEvent   m_ctx->_event_id(tagoffset) 0 ctrl [-]
    2021-08-25 23:48:06.423 INFO  [29968] [CManager::initEvent@248]  m_mode 3
    2021-08-25 23:48:06.428 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 9999 m_ni 65
    2021-08-25 23:48:06.429 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 9998 m_ni 65
    2021-08-25 23:48:06.429 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 9997 m_ni 65
    2021-08-25 23:48:06.429 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 9996 m_ni 65
    ...
    2021-08-25 23:48:16.041 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 67 m_ni 65
    2021-08-25 23:48:16.041 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 66 m_ni 65
    2021-08-25 23:48:16.041 FATAL [29968] [CWriter::writeStepPoint@207]  SKIP  unexpected record_id 65 m_ni 65
    2021-08-25 23:48:16.041 FATAL [29968] [NPY<T>::setValue@2965]  i 64 m_ni 0
    CG4Test: /home/blyth/opticks/npy/NPY.cpp:2966: void NPY<T>::setValue(int, int, int, int, T) [with T = double]: Assertion `in_range' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe878e387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-3.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libgcc-4.8.5-44.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-44.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-23.el7_9.x86_64 openssl-libs-1.0.2k-21.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe878e387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe878fa78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe87871a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe8787252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffef6f478e in NPY<double>::setValue (this=0x23356af0, i=64, j=0, k=0, l=0, value=0) at /home/blyth/opticks/npy/NPY.cpp:2966
    #5  0x00007fffef6f504a in NPY<double>::setQuad_ (this=0x23356af0, vec=..., i=64, j=0, k=0) at /home/blyth/opticks/npy/NPY.cpp:3257
    #6  0x00007ffff7b1c475 in CWriter::writeStepPoint_ (this=0x1d864e90, point=0xaadc350, photon=..., record_id=64) at /home/blyth/opticks/cfg4/CWriter.cc:301
    #7  0x00007ffff7b1c010 in CWriter::writeStepPoint (this=0x1d864e90, point=0xaadc350, flag=4096, material=1, last=false) at /home/blyth/opticks/cfg4/CWriter.cc:231
    #8  0x00007ffff7b13068 in CRecorder::WriteStepPoint (this=0x1d8650c0, point=0xaadc350, flag=4096, material=1, boundary_status=Undefined, last=false) at /home/blyth/opticks/cfg4/CRecorder.cc:755
    #9  0x00007ffff7b1262d in CRecorder::postTrackWriteSteps (this=0x1d8650c0) at /home/blyth/opticks/cfg4/CRecorder.cc:645
    #10 0x00007ffff7b109ef in CRecorder::postTrack (this=0x1d8650c0) at /home/blyth/opticks/cfg4/CRecorder.cc:213
    #11 0x00007ffff7b3dcae in CManager::postTrack (this=0x1d835580) at /home/blyth/opticks/cfg4/CManager.cc:349
    #12 0x00007ffff7b3dc1c in CManager::PostUserTrackingAction (this=0x1d835580, track=0x23522d60) at /home/blyth/opticks/cfg4/CManager.cc:317
    #13 0x00007ffff7b366a2 in CTrackingAction::PostUserTrackingAction (this=0xab28dc0, track=0x23522d60) at /home/blyth/opticks/cfg4/CTrackingAction.cc:79
    #14 0x00007ffff4940a1d in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #15 0x00007ffff4b76f61 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #16 0x00007ffff4e0ee87 in G4RunManager::ProcessOneEvent(int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #17 0x00007ffff4e080f3 in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #18 0x00007ffff4e07ebe in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #19 0x00007ffff7b3b299 in CG4::propagate (this=0xa934750) at /home/blyth/opticks/cfg4/CG4.cc:399
    #20 0x0000000000404556 in main (argc=2, argv=0x7fffffff65a8) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:76
    (gdb) 


The *CWriter* machinery is expecting to be informed at *BeginOfGenstep*, probably that only happening at BeginOfEvent::

    143 /**
    144 CWriter::BeginOfGenstep
    145 -------------------------
    146 
    147 Invoked from CRecorder::BeginOfGenstep, expands the buffers to accomodate the photons of this genstep.
    148 
    149 **/
    150 
    151 void CWriter::BeginOfGenstep()
    152 {   
    153     unsigned genstep_num_photons =  m_ctx._genstep_num_photons ;
    154     m_ni = expand(genstep_num_photons);
    155     
    156     LOG(LEVEL)
    157         << " m_ctx._gentype [" <<  m_ctx._gentype << "]" 
    158         << " m_ctx._genstep_index " << m_ctx._genstep_index
    159         << " m_ctx._genstep_num_photons " << m_ctx._genstep_num_photons
    160         << " m_ni " << m_ni
    161         ;
    162 
    163 
    164 }


CGenstepCollector::addGenstep needs to be called to prime the CWriter::

    283 /**
    284 CGenstepCollector::addGenstep
    285 -------------------------------
    286 
    287 Invoked from::
    288 
    289     CGenstepCollector::collectScintillationStep
    290     CGenstepCollector::collectCerenkovStep
    291     CGenstepCollector::collectMachineryStep
    292     CGenstepCollector::collectTorchGenstep    
    293 
    294 The automatic invokation of BeginOfGenstep from CGenstepCollector 
    295 is convenient for C+S gensteps but it is too early with input_photon 
    296 torch gensteps as the OpticksEvent is not yet created.  
    297 Instead the BeginOfGenstep for input photons is special case called 
    298 from CManager::BeginOfEventAction when input photons are detected 
    299 in CCtx::setEvent 
    300 
    301 **/
    302 
    303 CGenstep CGenstepCollector::addGenstep(unsigned numPhotons, char gentype)
    304 {
    305     unsigned genstep_index = getNumGensteps();
    306     unsigned photon_offset = getNumPhotons();
    307 
    308     CGenstep gs(genstep_index, numPhotons, photon_offset, gentype) ;
    309 
    310     LOG(LEVEL) << " gs.desc " << gs.desc() ;
    311 
    312     m_gs.push_back(gs);
    313     m_gs_photons.push_back(numPhotons);
    314     m_gs_offset.push_back(photon_offset);
    315     m_gs_type.push_back(gentype);
    316 
    317     m_photon_count += numPhotons ;
    318 
    319     CManager* mgr = CManager::Get();
    320     if(mgr && (gentype == 'C' || gentype == 'S'))   

    //// hmm : missed 'T' 

    321     {
    322         mgr->BeginOfGenstep(genstep_index, gentype, numPhotons, photon_offset);
    323     }
    324 
    325     return gs  ;
    326 }


CG4Test.cc is adding 'T' gensteps::

    051     CG4* g4 = new CG4(&hub) ;
     52     LOG(warning) << " post CG4 " ;
     53 
     54     g4->interactive();
     55 
     56     LOG(warning) << "  post CG4::interactive"  ;
     57 
     58     if(ok.isFabricatedGensteps())  // eg TORCH running
     59     {
     60         NPY<float>* gs = gen->getInputGensteps() ;
     61         unsigned numPhotons = G4StepNPY::CountPhotons(gs);
     62 
     63         LOG(error) << " setting gensteps " << gs << " numPhotons " << numPhotons ;
     64         char ctrl = '=' ;
     65         ok.createEvent(gs, ctrl);
     66 
     67         CGenstep cgs = g4->addGenstep(numPhotons, 'T' );
     68         LOG(info) << " cgs " << cgs.desc() ;
     69 
     70     }

    295 CGenstep CG4::addGenstep( unsigned num_photons, char gentype )
    296 {
    297     assert( m_collector );
    298     return m_collector->addGenstep( num_photons, gentype );
    299 }



::

    2021-08-26 02:07:28.525 INFO  [246640] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-08-26 02:07:28.526 FATAL [246640] [CWriter::expand@129]  Cannot expand as CWriter::initEvent has not been called, check CManager logging 


    O[blyth@localhost cfg4]$ export CManager=INFO
    O[blyth@localhost cfg4]$ gdb CG4Test 




::

    072 /**
     73 CWriter::initEvent
     74 -------------------
     75 
     76 Gets refs to the history, photons and records buffers from the event.
     77 When dynamic the records target is single item dynamic_records otherwise
     78 goes direct to the records_buffer.
     79 
     80 **/
     81 
     82 void CWriter::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
     83 {
     84     m_evt = evt ;
     85     assert(m_evt && m_evt->isG4());
     86 
     87     m_evt->setDynamic(1) ;
     88 
     89     LOG(LEVEL)
     90         << " _record_max " << m_ctx._record_max
     91         << " _bounce_max  " << m_ctx._bounce_max
     92         << " _steps_per_photon " << m_ctx._steps_per_photon
     93         << " num_g4event " << m_evt->getNumG4Event()
     94         ;
     95 
     96     m_history_buffer = m_evt->getSequenceData();  // ph : seqhis/seqmat
     97     m_photons_buffer = m_evt->getPhotonData();    // ox : final photon
     98     m_records_buffer = m_evt->getRecordData();    // rx :  step records
     99     m_deluxe_buffer  = m_evt->getDeluxeData();    // dx :  step records
    100 
    101     LOG(LEVEL) << desc() ;
    102 }

    117 /**
    118 CWriter::expand
    119 ----------------
    120 
    121 Invoked by CWriter::BeginOfGenstep
    122 
    123 
    124 **/
    125 unsigned CWriter::expand(unsigned gs_photons)
    126 {
    127     if(!m_history_buffer)
    128     {
    129         LOG(fatal) << " Cannot expand as CWriter::initEvent has not been called, check CManager logging " ;
    130         return 0 ;
    131     }
    132     assert( m_history_buffer );
    133     unsigned ni, ni1, ni2, ni3 ;
    134     ni = m_history_buffer->expand(gs_photons);
    135     ni1 = m_photons_buffer->expand(gs_photons);
    136     ni2 = m_records_buffer->expand(gs_photons);
    137     ni3 = m_deluxe_buffer->expand(gs_photons);
    138     assert( ni1 == ni && ni2 == ni && ni3 == ni );
    139     return ni ;
    140 }
    141 


    338 /**
    339 CG4::initEvent
    340 ----------------
    341 
    342 Invoked by CG4::propagate with the G4 OpticksEvent 
    343 
    344 **/
    345 
    346 void CG4::initEvent(OpticksEvent* evt)
    347 {
    348     LOG(LEVEL) << "[" ;
    349     m_generator->configureEvent(evt);
    350 
    351     // this should happen from CEventAction::BeginOfEventAction
    352     //m_manager->initEvent(evt); 
    353 
    354     LOG(LEVEL) << "]" ;
    355 }


Need to follow the pattern of G4OpticksRecorder and its CManager instance with CG4 playmng same role as G4OpticksRecorder.





CPropLib::addScintillatorMaterialProperties assert now FIXED : was misnaming LS to LS_ori due to only init m_original_domain in one GPropertMap ctor
------------------------------------------------------------------------------------------------------------------------------------------------------

::

    39/46 Test #39: CFG4Test.CGenstepSourceTest ...............Subprocess aborted***Exception:   2.32 sec
    2021-08-25 19:40:43.807 INFO  [93237] [OpticksHub::loadGeometry@283] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1
    2021-08-25 19:40:45.212 INFO  [93237] [OpticksHub::loadGeometry@315] ]
    2021-08-25 19:40:45.212 INFO  [93237] [Opticks::makeSimpleTorchStep@4218] [ts.setFrameTransform
    CGenstepSourceTest: /home/blyth/opticks/cfg4/CPropLib.cc:354: void CPropLib::addScintillatorMaterialProperties(G4MaterialPropertiesTable*, const char*): Assertion `scintillator && "non-zero reemission prob materials should has an associated raw scintillator"' failed.

    O[blyth@localhost opticks]$ gdb CMaterialTest 
    (gdb) r
    Starting program: /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    2021-08-25 19:45:43.569 INFO  [101555] [main@74] /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest
    2021-08-25 19:45:43.579 INFO  [101555] [OpticksHub::loadGeometry@283] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1
    2021-08-25 19:45:45.002 INFO  [101555] [OpticksHub::loadGeometry@315] ]
    2021-08-25 19:45:45.003 INFO  [101555] [Opticks::makeSimpleTorchStep@4218] [ts.setFrameTransform
    2021-08-25 19:45:45.003 INFO  [101555] [main@82] /data/blyth/junotop/ExternalLibs/opticks/head/lib/CMaterialTest convert 
    CMaterialTest: /home/blyth/opticks/cfg4/CPropLib.cc:354: void CPropLib::addScintillatorMaterialProperties(G4MaterialPropertiesTable*, const char*): Assertion `scintillator && "non-zero reemission prob materials should has an associated raw scintillator"' failed.

    (gdb) bt
    #3  0x00007fffe8788252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7ad0e56 in CPropLib::addScintillatorMaterialProperties (this=0xa8facc0, mpt=0xa925420, name=0x712bd0 "LS") at /home/blyth/opticks/cfg4/CPropLib.cc:354
    #5  0x00007ffff7ad09bd in CPropLib::makeMaterialPropertiesTable (this=0xa8facc0, ggmat=0x712ad0) at /home/blyth/opticks/cfg4/CPropLib.cc:276
    #6  0x00007ffff7ae2563 in CMaterialLib::convertMaterial (this=0xa8facc0, kmat=0x712ad0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:261
    #7  0x00007ffff7ae18bb in CMaterialLib::convert (this=0xa8facc0) at /home/blyth/opticks/cfg4/CMaterialLib.cc:154
    #8  0x0000000000403eaf in main (argc=1, argv=0x7fffffffa188) at /home/blyth/opticks/cfg4/tests/CMaterialTest.cc:84
    (gdb) 


::

    351 void CPropLib::addScintillatorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name )
    352 {
    353     GPropertyMap<double>* scintillator = m_sclib->getRaw(name);
    354     assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    355     LOG(LEVEL)
    356         << " found corresponding scintillator from sclib "
    357         << " name " << name
    358         << " keys " << scintillator->getKeysString()
    359         ;
    360 
    361     bool keylocal = false ;
    362     bool constant = false ;
    363     addProperties(mpt, scintillator, "SLOWCOMPONENT,FASTCOMPONENT", keylocal, constant);
    364     addProperties(mpt, scintillator, "SCINTILLATIONYIELD,RESOLUTIONSCALE,YIELDRATIO,FASTTIMECONSTANT,SLOWTIMECONSTANT", keylocal, constant ); // this used constant=true formerly
    365 
    366     // NB the above skips prefixed versions of the constants: Alpha, 
    367     //addProperties(mpt, scintillator, "ALL",          keylocal=false, constant=true );
    368 }



Curious. CMaterialTest not failing on Darwin. Must be from whats in geocache.

::

   O[blyth@localhost cfg4]$ CMaterialLib=INFO CMaterialTest 


::

     431 void X4PhysicalVolume::createScintillatorGeant4InterpolatedICDF()
     432 {
     433     unsigned num_scint = m_sclib->getNumRawOriginal() ;
     434     if( num_scint == 0 ) return ;
     435     //assert( num_scint == 1 ); 
     436 
     437     typedef GPropertyMap<double> PMAP ;
     438     PMAP* pmap_en = m_sclib->getRawOriginal(0u);
     439     assert( pmap_en );
     440     assert( pmap_en->hasOriginalDomain() );
     441 
     442     NPY<double>* slow_en = pmap_en->getProperty("SLOWCOMPONENT")->makeArray();
     443     NPY<double>* fast_en = pmap_en->getProperty("FASTCOMPONENT")->makeArray();
     444 
     445     //slow_en->save("/tmp/slow_en.npy"); 
     446     //fast_en->save("/tmp/fast_en.npy"); 
     447 
     448     X4Scintillation xs(slow_en, fast_en);
     449 
     450     unsigned num_bins = 4096 ;
     451     unsigned hd_factor = 20 ;
     452     const char* material_name = pmap_en->getName() ;
     453 
     454     NPY<double>* g4icdf = xs.createGeant4InterpolatedInverseCDF(num_bins, hd_factor, material_name ) ;
     455 
     456     LOG(info)
     457         << " num_scint " << num_scint
     458         << " slow_en " << slow_en->getShapeString()
     459         << " fast_en " << fast_en->getShapeString()
     460         << " num_bins " << num_bins
     461         << " hd_factor " << hd_factor
     462         << " material_name " << material_name
     463         << " g4icdf " << g4icdf->getShapeString()
     464         ;
     465 
     466     m_sclib->setGeant4InterpolatedICDF(g4icdf);   // trumps legacyCreateBuffer
     467     m_sclib->close();   // creates and sets "THE" buffer 
     468 }
     469 



::

    epsilon:extg4 blyth$ opticks-f getRawOriginal
    ./extg4/X4PhysicalVolume.cc:    PMAP* pmap_en = m_sclib->getRawOriginal(0u); 
    ./ggeo/GPropertyLib.cc:GPropertyMap<double>* GPropertyLib::getRawOriginal(unsigned index) const 
    ./ggeo/GPropertyLib.cc:GPropertyMap<double>* GPropertyLib::getRawOriginal(const char* shortname) const 
    ./ggeo/GPropertyLib.hh:        GPropertyMap<double>* getRawOriginal(unsigned index) const ;
    ./ggeo/GPropertyLib.hh:        GPropertyMap<double>* getRawOriginal(const char* shortname) const ;

    epsilon:opticks blyth$ opticks-f addRawOriginal
    ./extg4/X4PhysicalVolume.cc:        m_sclib->addRawOriginal(pmap);      
    ./extg4/X4MaterialTable.cc:        m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib
    ./ggeo/GPropertyLib.cc:void GPropertyLib::addRawOriginal(GPropertyMap<double>* pmap)
    ./ggeo/GPropertyLib.hh:        void                  addRawOriginal(GPropertyMap<double>* pmap);
    epsilon:opticks blyth$ 



::

     388 void X4PhysicalVolume::collectScintillatorMaterials()
     389 {   
     390     assert( m_sclib ); 
     391     std::vector<GMaterial*>  scintillators_raw = m_mlib->getRawMaterialsWithProperties(SCINTILLATOR_PROPERTIES, ',' );
     392     
     393     typedef GPropertyMap<double> PMAP ;  
     394     std::vector<PMAP*> raw_energy_pmaps ;  
     395     m_mlib->findRawOriginalMapsWithProperties( raw_energy_pmaps, SCINTILLATOR_PROPERTIES, ',' );
     396     
     397     bool consistent = scintillators_raw.size() == raw_energy_pmaps.size()  ;
     398     if(!consistent)
     399         LOG(fatal) 
     400             << " scintillators_raw.size " << scintillators_raw.size()
     401             << " raw_energy_pmaps.size " << raw_energy_pmaps.size()
     402             ;
     403     
     404     assert( consistent ); 
     405     unsigned num_scint = scintillators_raw.size() ;
     406     
     407     if(num_scint == 0)
     408     {   
     409         LOG(LEVEL) << " found no scintillator materials  " ;
     410         return ;
     411     }
     412     
     413     LOG(info) << " found " << num_scint << " scintillator materials  " ;
     414     
     415     // wavelength domain 
     416     for(unsigned i=0 ; i < num_scint ; i++)
     417     {   
     418         GMaterial* mat_ = scintillators_raw[i] ;
     419         PMAP* mat = dynamic_cast<PMAP*>(mat_);
     420         m_sclib->addRaw(mat);
     421     }
     422     
     423     // original energy domain 
     424     for(unsigned i=0 ; i < num_scint ; i++)
     425     {   
     426         PMAP* pmap = raw_energy_pmaps[i] ;
     427         m_sclib->addRawOriginal(pmap);
     428     }
     429 }




FIXED : was an uninitialized m_domain_original : causing unexpected : GScintillatorLib.getNumRaw  0 GScintillatorLib.getNumRawOriginal  1  : should be the same
------------------------------------------------------------------------------------------------------------------------------------------------------------------

::

    2021-08-25 22:14:49.023 INFO  [333605] [CMaterialLib::convertMaterial@239]  name LS sname LS materialIndex 0
    2021-08-25 22:14:49.025 FATAL [333605] [CPropLib::addScintillatorMaterialProperties@358]  FAILED to find material in m_sclib (GScintillatorLib) with name LS
    2021-08-25 22:14:49.025 INFO  [333605] [GScintillatorLib::Summary@51] CPropLib::addScintillatorMaterialProperties GScintillatorLib.getNumRaw  0 GScintillatorLib.getNumRawOriginal  1
    2021-08-25 22:14:49.025 INFO  [333605] [GPropertyLib::dumpRaw@937] CPropLib::addScintillatorMaterialProperties
    CMaterialTest: /home/blyth/opticks/cfg4/CPropLib.cc:361: void CPropLib::addScintillatorMaterialProperties(G4MaterialPropertiesTable*, const char*): Assertion `scintillator && "non-zero reemission prob materials should has an associated raw scintillator"' failed.
    Aborted (core dumped)
    O[blyth@localhost cfg4]$ 


geocache-kcd::

    O[blyth@localhost 1]$ cd GScintillatorLib
    O[blyth@localhost GScintillatorLib]$ l
    total 112
      4 -rw-rw-r--.  1 blyth blyth   120 Aug 17 16:45 GScintillatorLib.json
    100 -rw-rw-r--.  1 blyth blyth 98384 Aug 17 16:45 GScintillatorLib.npy
      4 drwxrwxr-x. 13 blyth blyth  4096 Aug 17 16:44 ..
      4 drwxrwxr-x.  2 blyth blyth  4096 Jul  7 20:52 LS_ori
      0 drwxrwxr-x.  3 blyth blyth    77 Jul  7 20:52 .
    O[blyth@localhost GScintillatorLib]$ 

Darwin, geocache-kcd::

    epsilon:1 blyth$ cd GScintillatorLib/
    epsilon:GScintillatorLib blyth$ l
    total 208
      0 drwxr-xr-x  17 blyth  staff    544 Jul  7 17:26 ..
      0 drwxr-xr-x  34 blyth  staff   1088 Jul  7 17:26 LS_ori
      0 drwxr-xr-x   6 blyth  staff    192 Jul  7 17:26 .
      0 drwxr-xr-x  34 blyth  staff   1088 Jul  7 17:26 LS
    200 -rw-r--r--   1 blyth  staff  98384 Jul  7 17:26 GScintillatorLib.npy
      8 -rw-r--r--   1 blyth  staff    120 Jul  7 17:26 GScintillatorLib.json
    epsilon:GScintillatorLib blyth$ 



::

    105 void X4MaterialTable::init()
    106 {
    107     unsigned num_input_materials = m_input_materials.size() ;
    108 
    109     LOG(LEVEL) << ". G4 nmat " << num_input_materials ;
    110 
    111     for(unsigned i=0 ; i < num_input_materials ; i++)
    112     {
    113         G4Material* material = m_input_materials[i] ;
    114         G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    115 
    116         if( mpt == NULL )
    117         {
    118             LOG(error) << "PROCEEDING TO convert material with no mpt " << material->GetName() ;
    119             // continue ;  
    120         }
    121         else
    122         {
    123             LOG(LEVEL) << " converting material with mpt " <<  material->GetName() ;
    124         }
    125 
    126         //char mode_oldstandardized = 'S' ;
    127         char mode_g4interpolated = 'G' ;
    128         GMaterial* mat = X4Material::Convert( material, mode_g4interpolated );
    129         if(mat->hasProperty("EFFICIENCY")) m_materials_with_efficiency.push_back(material);
    130         m_mlib->add(mat) ;
    131 
    132         char mode_asis_nm = 'A' ;
    133         GMaterial* rawmat = X4Material::Convert( material, mode_asis_nm );
    134         m_mlib->addRaw(rawmat) ;
    135 
    136         char mode_asis_en = 'E' ;
    137         GMaterial* rawmat_en = X4Material::Convert( material, mode_asis_en );
    138         GPropertyMap<double>* pmap_rawmat_en = dynamic_cast<GPropertyMap<double>*>(rawmat_en) ;
    139         m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib
    140 
    141 
    142     }
    143 }



::

    tds3 onlt LS_ori is appearing 


    2021-08-25 22:36:30.378 INFO  [365931] [GPropertyLib::saveToCache@553] ]
    2021-08-25 22:36:30.378 INFO  [365931] [GPropertyLib::saveToCache@509]  dir /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GSurfaceLib name GSurfaceLibOptical.npy type GSurfaceLib
    2021-08-25 22:36:30.378 INFO  [365931] [GPropertyLib::saveToCache@531] [
    2021-08-25 22:36:30.379 INFO  [365931] [GPropertyLib::saveToCache@553] ]
    2021-08-25 22:36:30.379 INFO  [365931] [GPropertyLib::saveRaw@953] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw 1
    2021-08-25 22:36:30.381 INFO  [365931] [GPropertyLib::saveRaw@959] ]
    2021-08-25 22:36:30.381 INFO  [365931] [GPropertyLib::saveRawOriginal@966] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw_original 1
    2021-08-25 22:36:30.394 INFO  [365931] [GPropertyLib::saveRawOriginal@972] ]
    2021-08-25 22:36:30.394 INFO  [365931] [GPropertyLib::saveToCache@531] [
    2021-08-25 22:36:30.394 INFO  [365931] [GPropertyLib::saveToCache@553] ]
    2021-08-25 22:36:30.395 INFO  [365931] [GPropertyLib::saveToCache@509]  dir /home/blyth/.


Seems are not properly initializing m_original_domain, causing misnaming to LS_ori for both raw and raw_original when should be LS and LS_ori::

    2021-08-25 23:03:14.858 INFO  [410087] [GPropertyLib::saveToCache@531] [
    2021-08-25 23:03:14.859 INFO  [410087] [GPropertyLib::saveToCache@553] ]
    2021-08-25 23:03:14.859 INFO  [410087] [GPropertyLib::saveRaw@953] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw 1
    2021-08-25 23:03:14.859 INFO  [410087] [GPropertyMap<T>::save@1084]  save shortname (+_ori?) [LS_ori] m_original_domain 90
    2021-08-25 23:03:14.861 INFO  [410087] [GPropertyLib::saveRaw@959] ]
    2021-08-25 23:03:14.861 INFO  [410087] [GPropertyLib::saveRawOriginal@966] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw_original 1
    2021-08-25 23:03:14.861 INFO  [410087] [GPropertyMap<T>::save@1084]  save shortname (+_ori?) [LS_ori] m_original_domain 1
    2021-08-25 23:03:14.874 INFO  [410087] [GPropertyLib::saveRawOriginal@972] ]
    2021-08-25 23:03:14.874 INFO  [410087] [GPropertyLib::saveToCache@531] [


Fixed that, was only initializing in one of the three ctors::

    2021-08-25 23:07:47.292 INFO  [418537] [GPropertyLib::saveToCache@553] ]
    2021-08-25 23:07:47.292 INFO  [418537] [GPropertyLib::saveToCache@509]  dir /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GSurfaceLib name GSurfaceLibOptical.npy type GSurfaceLib
    2021-08-25 23:07:47.292 INFO  [418537] [GPropertyLib::saveToCache@531] [
    2021-08-25 23:07:47.293 INFO  [418537] [GPropertyLib::saveToCache@553] ]
    2021-08-25 23:07:47.293 INFO  [418537] [GPropertyLib::saveRaw@953] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw 1
    2021-08-25 23:07:47.293 INFO  [418537] [GPropertyMap<T>::save@1085]  save shortname (+_ori?) [LS] m_original_domain 0
    2021-08-25 23:07:47.293 INFO  [418537] [BFile::preparePath@836] created directory /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib/LS
    2021-08-25 23:07:47.299 INFO  [418537] [GPropertyLib::saveRaw@959] ]
    2021-08-25 23:07:47.299 INFO  [418537] [GPropertyLib::saveRawOriginal@966] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GScintillatorLib num_raw_original 1
    2021-08-25 23:07:47.299 INFO  [418537] [GPropertyMap<T>::save@1085]  save shortname (+_ori?) [LS_ori] m_original_domain 1
    2021-08-25 23:07:47.301 INFO  [418537] [GPropertyLib::saveRawOriginal@972] ]
    2021-08-25 23:07:47.301 INFO  [418537] [GPropertyLib::saveToCache@531] [
    2021-08-25 23:07:47.302 INFO  [418537] [GPropertyLib::saveToCache@553] ]
    2021-08-25 23:07:47.302 INFO  [418537] [GPropertyLib::saveToCache@509]  dir /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/b8bc31e2cdf88b66e3dfa9afd5ac1f2b/1/GBndLib name GBndLibIndex.npy type GBndLib



X4 GDML tempStr fails : fixed by decoupling from Geant4 so dont have to vary by Geant4 version
-----------------------------------------------------------------------------------------------------


::

    .     Start 18: ExtG4Test.X4CSGTest
    18/31 Test #18: ExtG4Test.X4CSGTest .....................................***Exception: SegFault  0.13 sec
          Start 20: ExtG4Test.X4GDMLParserTest
    20/31 Test #20: ExtG4Test.X4GDMLParserTest ..............................***Exception: SegFault  0.14 sec
    2021-08-25 18:36:11.175 FATAL [436528] [Opticks::envkey@345]  --allownokey option prevents key checking : this is for debugging of geocache creation 
    2021-08-25 18:36:11.179 FATAL [436528] [OpticksResource::init@122]  CAUTION : are allowing no key 

          Start 21: ExtG4Test.X4GDMLBalanceTest
    21/31 Test #21: ExtG4Test.X4GDMLBalanceTest .............................***Exception: SegFault  0.15 sec



::

    (gdb) f 12
    #12 0x00000000004035cd in main (argc=1, argv=0x7fffffffa428) at /home/blyth/opticks/extg4/tests/X4CSGTest.cc:59
    59	    X4CSG::GenerateTest( solid, &ok, prefix, lvidx ) ;
    (gdb) f 11
    #11 0x00007ffff7b49d86 in X4CSG::GenerateTest (solid=0x6bc010, ok=0x7fffffffa0f0, prefix=0x40617b "$TMP/extg4/X4CSGTest", lvidx=1) at /home/blyth/opticks/extg4/X4CSG.cc:78
    78	    X4CSG xcsg(solid, ok);
    (gdb) f 10
    #10 0x00007ffff7b4a202 in X4CSG::X4CSG (this=0x7fffffff9cd0, solid_=0x6bc010, ok_=0x7fffffffa0f0) at /home/blyth/opticks/extg4/X4CSG.cc:131
    131	    index(-1)
    (gdb) f 9
    #9  0x00007ffff7b68ddb in X4GDMLParser::ToString (solid=0x6bc010, refs=false) at /home/blyth/opticks/extg4/X4GDMLParser.cc:57
    57	    X4GDMLParser parser(refs) ; 
    (gdb) f 8
    #8  0x00007ffff7b68e5c in X4GDMLParser::X4GDMLParser (this=0x7fffffff9c50, refs=false) at /home/blyth/opticks/extg4/X4GDMLParser.cc:69
    69	    writer = new X4GDMLWriteStructure(refs) ; 
    (gdb) f 7
    #7  0x00007ffff7b69942 in X4GDMLWriteStructure::X4GDMLWriteStructure (this=0x712ac0, refs=false) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:35
    35	    init(refs); 
    (gdb) f 6
    #6  0x00007ffff7b69a5f in X4GDMLWriteStructure::init (this=0x712ac0, refs=false) at /home/blyth/opticks/extg4/X4GDMLWriteStructure.cc:63
    63	   xercesc::XMLString::transcode("LS", tempStr, 9999);
    (gdb) p tempStr
    $1 = (XMLCh *) 0x0
    (gdb) 



1042::

    epsilon:gdml blyth$ pwd
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml
    epsilon:gdml blyth$ 

    epsilon:gdml blyth$ find . -type f  -exec grep -H tempStr {} \;
    ./include/G4GDMLWrite.hh:    XMLCh tempStr[10000];
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(value,tempStr,9999);
    ./src/G4GDMLWrite.cc:   att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(str,tempStr,9999);
    ./src/G4GDMLWrite.cc:   att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,9999);
    ./src/G4GDMLWrite.cc:   return doc->createElement(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("LS", tempStr, 9999);
    ./src/G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("Range", tempStr, 9999);
    ./src/G4GDMLWrite.cc:     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:   xercesc::XMLString::transcode("gdml", tempStr, 9999);
    ./src/G4GDMLWrite.cc:   doc = impl->createDocument(0,tempStr,0);
    epsilon:gdml blyth$ 




    128 
    129   protected:
    130 
    131     G4String SchemaLocation;
    132     static G4bool addPointerToName;
    133     xercesc::DOMDocument* doc;
    134     xercesc::DOMElement* extElement;
    135     xercesc::DOMElement* userinfoElement;
    136     XMLCh tempStr[10000];
    137     G4GDMLAuxListType auxList;
    138 };
    139 




1070 still the same::

    epsilon:gdml blyth$ find . -type f -exec grep -H tempStr {} \;
    ./include/G4GDMLWrite.hh:    XMLCh tempStr[10000];
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(value, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(str, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  att->setValue(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode(name, tempStr, 9999);
    ./src/G4GDMLWrite.cc:  return doc->createElement(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("LS", tempStr, 9999);
    ./src/G4GDMLWrite.cc:  xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("Range", tempStr, 9999);
    ./src/G4GDMLWrite.cc:    xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    ./src/G4GDMLWrite.cc:  xercesc::XMLString::transcode("gdml", tempStr, 9999);
    ./src/G4GDMLWrite.cc:  doc                       = impl->createDocument(0, tempStr, 0);
    epsilon:gdml blyth$ pwd
    /usr/local/opticks_externals/g4_1070.build/geant4.10.07/source/persistency/gdml

The tempStr disappears at some point after 1070.

Old way with fixed size tempStr::

    137 xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
    138                                             const G4String& value)
    139 {
    140    xercesc::XMLString::transcode(name,tempStr,9999);
    141    xercesc::DOMAttr* att = doc->createAttribute(tempStr);
    142    xercesc::XMLString::transcode(value,tempStr,9999);
    143    att->setValue(tempStr);
    144    return att;
    145 }


New way::

    https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/src/G4GDMLWrite.cc

    xercesc::DOMAttr* G4GDMLWrite::NewAttribute(const G4String& name,
                                                const G4String& value)
    {
      XMLCh* tempStr = NULL;
      tempStr = xercesc::XMLString::transcode(name);
      xercesc::DOMAttr* att = doc->createAttribute(tempStr);
      xercesc::XMLString::release(&tempStr);

      tempStr = xercesc::XMLString::transcode(value);
      att->setValue(tempStr);
      xercesc::XMLString::release(&tempStr);

      return att;
    }



* https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/include/G4GDMLWrite.hh



::

    epsilon:opticks blyth$ git add . 
    epsilon:opticks blyth$ git commit -m "try to avoid needing to change X4GDMLWriteStructure with Geant4 version by using XMLCh local_tempStr[10000] " 
    [master 29a47cb7d] try to avoid needing to change X4GDMLWriteStructure with Geant4 version by using XMLCh local_tempStr[10000]
     3 files changed, 207 insertions(+), 7 deletions(-)
     create mode 100644 notes/issues/opticks-t-fails-aug-2021-13-of-493.rst
    epsilon:opticks blyth$ git push 
    Counting objects: 8, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (8/8), done.
    Writing objects: 100% (8/8), 3.00 KiB | 3.00 MiB/s, done.
    Total 8 (delta 6), reused 0 (delta 0)
    To bitbucket.org:simoncblyth/opticks.git
       31a2c9e75..29a47cb7d  master -> master
    epsilon:opticks blyth$ 



