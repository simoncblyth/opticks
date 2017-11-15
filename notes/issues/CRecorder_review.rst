CRecorder_review
====================


Simplifications
-----------------

* split off CRecorderWriter 
* introduce shared struct CG4Ctx : to avoid playing pass the parcel







Collateral Damage
-------------------

* looks like CRec/m_crec not getting cleared

::


    tboolean-;tboolean-sphere --okg4 -D --dbgrec 


    2017-11-14 20:40:53.911 INFO  [4972251] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-11-14 20:40:53.924 WARN  [4972251] [OpPointFlag@273]  OpPointFlag ZERO   proceesDefinedStep? NoProc stage START status Undefined
    Assertion failed: (0), function RecordStepPoint, file /Users/blyth/opticks/cfg4/CRecorder.cc, line 822.
    Process 60135 stopped
    * thread #1: tid = 0x4bdedb, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x4bdedb, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001043b1d86 libcfg4.dylib`CRecorder::RecordStepPoint(this=0x0000000112560000, point=0x00000001205e1bb0, flag=0, material=2, boundary_status=Undefined, label=0x00000001043fa4e5) + 278 at CRecorder.cc:822
        frame #5: 0x00000001043b306e libcfg4.dylib`CRecorder::CannedWriteSteps(this=0x0000000112560000) + 1806 at CRecorder.cc:713
        frame #6: 0x00000001043b3297 libcfg4.dylib`CRecorder::posttrack(this=0x0000000112560000) + 39 at CRecorder.cc:773
        frame #7: 0x00000001043e3894 libcfg4.dylib`CTrackingAction::PostUserTrackingAction(this=0x000000011255e810, track=0x00000001205e0390) + 548 at CTrackingAction.cc:159
        frame #8: 0x000000010518f7d1 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000112522530, apValueG4Track=<unavailable>) + 1009 at G4TrackingManager.cc:140
        frame #9: 0x00000001050e7727 libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001125224a0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #10: 0x0000000105069611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109716f30, i_event=0) + 49 at G4RunManager.cc:399
        frame #11: 0x00000001050694db libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109716f30, n_event=60, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #12: 0x0000000105068913 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109716f30, n_event=60, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #13: 0x00000001043e67f3 libcfg4.dylib`CG4::propagate(this=0x0000000109716df0) + 1667 at CG4.cc:333
        frame #14: 0x00000001044de28a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe460) + 538 at OKG4Mgr.cc:82
        frame #15: 0x00000001000132fa OKG4Test`main(argc=27, argv=0x00007fff5fbfe548) + 1498 at OKG4Test.cc:57
        frame #16: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #17: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) f 5
    frame #5: 0x00000001043b306e libcfg4.dylib`CRecorder::CannedWriteSteps(this=0x0000000112560000) + 1806 at CRecorder.cc:713
       710          if(i == 0)
       711          {
       712              m_step_action |= PRE_SAVE ; 
    -> 713              done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );  
       714              if(done) m_step_action |= PRE_DONE ; 
       715  
       716              if(!done)
    (lldb) p preFlag
    (unsigned int) $0 = 0
    (lldb) 




Candle While Make Simplifications
-----------------------------------

* note the seqmat SR bug 

::

    simon:cfg4 blyth$ tboolean-;tboolean-sphere-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-sphere --tag 1
    ok.smry 1 
    [2017-11-14 18:08:09,657] p53147 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-sphere c2max 2.0 ipython False 
    [2017-11-14 18:08:09,657] p53147 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-14 18:08:10,415] p53147 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-14 18:08:10,418] p53147 {/Users/blyth/opticks/ana/ab.py:135} INFO - AB.init_point START
    [2017-11-14 18:08:10,418] p53147 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-sphere)  None 0 
    A tboolean-sphere/torch/  1 :  20171114-1807 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/1/fdom.npy 
    B tboolean-sphere/torch/ -1 :  20171114-1807 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-sphere/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum/perfectSpecularSurface//Pyrex
    /tmp/blyth/opticks/tboolean-sphere--
    .                seqhis_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000         5.42/5 =  1.08  (pval:0.367 prob:0.633)  
    0000     482087    482071             0.00  TO SA
    0001     117397    117417             0.00  TO SR SA
    0002        403       385             0.41  TO SC SA
    0003         35        56             4.85  TO AB
    0004         35        34             0.01  TO SC SR SA
    0005         33        30             0.14  TO SR SC SA
    0006          5         6             0.00  TO SR AB
    0007          5         1             0.00  TO SR SC SR SA
    .                             600000    600000         5.42/5 =  1.08  (pval:0.367 prob:0.633)  
    .                pflags_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000         5.72/4 =  1.43  (pval:0.221 prob:0.779)  
    0000     482087    482071             0.00  TO|SA
    0001     117397    117417             0.00  TO|SR|SA
    0002        403       385             0.41  TO|SA|SC
    0003         73        65             0.46  TO|SR|SA|SC
    0004         35        56             4.85  TO|AB
    0005          5         6             0.00  TO|SR|AB
    .                             600000    600000         5.72/4 =  1.43  (pval:0.221 prob:0.779)  
    .                seqmat_ana  1:tboolean-sphere   -1:tboolean-sphere        c2        ab        ba 
    .                             600000    600000    234083.20/4 = 58520.80  (pval:0.000 prob:1.000)  
    0000     482087    482071             0.00  Vm Rk
    0001     117800       385        116650.02  Vm Vm Rk
    0002          0    117417        117417.00  Py Vm Rk
    0003         68        34            11.33  Vm Vm Vm Rk
    0004         35        56             4.85  Vm Vm
    0005          0        30             0.00  Py Vm Vm Rk
    0006          0         6             0.00  Py Vm Vm
    0007          5         0             0.00  Vm Vm Vm
    0008          5         0             0.00  Vm Vm Vm Vm Rk
    0009          0         1             0.00  Py Vm Vm Vm Rk
    .                             600000    600000    234083.20/4 = 58520.80  (pval:0.000 prob:1.000)  
                  /tmp/blyth/opticks/evt/tboolean-sphere/torch/1 9fe099d814f9ed5a1a4fa784110047af 7a3ebc21c6f795d198b9ee1494917b32  600000    -1.0000 INTEROP_MODE 
    {u'emitconfig': u'photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1, u'poly': u'IM'}
    [2017-11-14 18:08:10,422] p53147 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:cfg4 blyth$ 



Observations
---------------

* machinery is overcomplicated due to passing things around, 
  like photon_id/record_id etc..
  better to have a shared context struct  

::

    113 CG4::CG4(OpticksHub* hub)
    114    :
    115      m_hub(hub),
    116      m_ok(m_hub->getOpticks()),
    117      m_run(m_ok->getRun()),
    118      m_cfg(m_ok->getCfg()),
    119      m_physics(new CPhysics(this)),
    120      m_runManager(m_physics->getRunManager()),
    121      m_geometry(new CGeometry(m_hub)),
    122      m_hookup(m_geometry->hookup(this)),
    123      m_mlib(m_geometry->getMaterialLib()),
    124      m_detector(m_geometry->getDetector()),
    125      m_generator(new CGenerator(m_hub, this)),
    126      m_dynamic(m_generator->isDynamic()),
    127      m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
    128      m_recorder(new CRecorder(m_ok, m_geometry, m_dynamic)),
    129      m_steprec(new CStepRec(m_ok, m_dynamic)),
    130      m_visManager(NULL),
    131      m_uiManager(NULL),
    132      m_ui(NULL),
    133      m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
    134      m_sa(new CSteppingAction(this, m_generator->isDynamic())),
    135      m_ta(new CTrackingAction(this)),
    136      m_ra(new CRunAction(m_hub)),
    137      m_ea(new CEventAction(this)),
    138      m_initialized(false)
    139 {
    140      OK_PROFILE("CG4::CG4");
    141      init();
    142 }

 




use of CRecorder/m_recorder
-------------------------------

::

    simon:cfg4 blyth$ grep  m_recorder\-\>  *.*
    CG4.cc:    m_recorder->postinitialize();  
    CG4.cc:    m_recorder->initEvent(evt);
    CGunSource.cc:    //m_recorder->RecordPrimaryVertex(vertex);
    CSteppingAction.cc:   m_verbosity(m_recorder->getVerbosity()),
    CSteppingAction.cc:    m_recorder->setPhotonId(m_photon_id);   
    CSteppingAction.cc:    m_recorder->setEventId(m_event_id);
    CSteppingAction.cc:    int record_max = m_recorder->getRecordMax() ;
    CSteppingAction.cc:        done = m_recorder->Record(m_step, m_step_id, m_record_id, m_debug, m_other, boundary_status, stage);
    CSteppingAction.cc:    m_recorder->report(msg);
    CTrackingAction.cc:        m_recorder->posttrack();
    simon:cfg4 blyth$ 

    simon:cfg4 blyth$ grep setRecordId *.*
    CRecorder.cc:void CRecorder::setRecordId(int record_id, bool dbg, bool other)
    CRecorder.cc:    setRecordId(record_id, dbg, other );
    CRecorder.hh:        void setRecordId(int record_id, bool dbg, bool other);
    CSteppingAction.cc:void CSteppingAction::setRecordId(int record_id, bool dbg, bool other)
    CSteppingAction.hh:    void setRecordId(int photon_id, bool debug, bool other);
    CSteppingAction.hh:    // set by setRecordId
    CTrackingAction.cc:    setRecordId(record_id);
    CTrackingAction.cc:void CTrackingAction::setRecordId(int record_id )
    CTrackingAction.cc:    m_sa->setRecordId(record_id, _debug, other);
    CTrackingAction.hh:    void setRecordId(int record_id);
    CTrackingAction.hh:    // setRecordId




CTrackingAction::PreUserTrackingAction gets ball rolling with setTrack
------------------------------------------------------------------------


::

    217 void CTrackingAction::PreUserTrackingAction(const G4Track* track)
    218 {
    219    // TODO: move to CEventAction
    220    // const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;
    221    // setEvent(event);
    222 
    223     setTrack(track);
    224 
    225     LOG(trace) << "CTrackingAction::PreUserTrackingAction"
    226               << brief()
    227                ;
    228 }
    229 
    230 void CTrackingAction::PostUserTrackingAction(const G4Track* track)
    231 {
    232     int track_id = CTrack::Id(track) ;
    233     assert( track_id == m_track_id );
    234     assert( track == m_track );
    235 
    236     LOG(trace) << "CTrackingAction::PostUserTrackingAction"
    237               << brief()
    238               ;
    239 
    240     if(m_optical)
    241     {
    242         m_recorder->posttrack();
    243     }
    244 }


m_record_id : crucial absolute record index across multiple G4Event
---------------------------------------------------------------------

* enables multiple G4Event to feed into a single "OpticksEvent"



::

    144 void CTrackingAction::setPhotonId(int photon_id, bool reemtrack)
    145 {
    146     m_photon_id = photon_id ;    // NB photon_id continues reemission photons
    147     m_reemtrack = reemtrack ;
    148 
    149     m_sa->setPhotonId(m_photon_id, m_reemtrack);
    150 
    151     int record_id = m_photons_per_g4event*m_event_id + m_photon_id ;
    152     setRecordId(record_id);
    153 
    154     if(m_dump) dump("CTrackingAction::setPhotonId");
    155 }

    157 void CTrackingAction::setRecordId(int record_id )
    158 {
    159     m_record_id = record_id ;
    160 
    161     bool _debug = m_ok->isDbgPhoton(record_id) ; // from option: --dindex=1,100,1000,10000 
    162     setDebug(_debug);
    163 
    164     bool other = m_ok->isOtherPhoton(record_id) ; // from option: --oindex=1,100,1000,10000 
    165     setOther(other);
    166 
    167     m_dump = m_debug || m_other ;
    168 
    169     m_sa->setRecordId(record_id, _debug, other);
    170 }





