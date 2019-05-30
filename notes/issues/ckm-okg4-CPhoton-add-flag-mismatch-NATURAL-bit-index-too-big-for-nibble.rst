ckm-okg4-CPhoton-add-flag-mismatch-NATURAL-bit-index-too-big-for-nibble
=========================================================================


Context :doc:`ckm-okg4-material-rindex-mismatch`

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }


ckm-okg4::

    2019-05-30 15:17:15.149 INFO  [445621] [SLog::operator@28]  ) OKPropagator::OKPropagator  DONE
    2019-05-30 15:17:15.149 INFO  [445621] [SLog::operator@28]  ) OKG4Mgr::OKG4Mgr  DONE
    2019-05-30 15:17:15.151 INFO  [445621] [OpticksRun::setGensteps@148] genstep 1,6,4
    2019-05-30 15:17:15.152 INFO  [445621] [CG4::propagate@304] Evt /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1 20190530_151715 /home/blyth/local/opticks/lib/OKG4Test  genstep 1,6,4 nopstep 0,4,4 photon 221,4,4 source NULL record 221,10,2,4 phosel 221,1,4 recsel 221,10,1,4 sequence 221,1,2 seed 221,1,1 hit 0,4,4
    2019-05-30 15:17:15.152 INFO  [445621] [CG4::propagate@322] CG4::propagate(0) /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1
    2019-05-30 15:17:15.152 INFO  [445621] [CGenerator::configureEvent@104] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    2019-05-30 15:17:15.152 INFO  [445621] [CG4Ctx::initEvent@150] CG4Ctx::initEvent _record_max (numPhotons from genstep summation) 221 photons_per_g4event 0 steps_per_photon 10 typ natural gen 65536 SourceType NATURAL
    2019-05-30 15:17:15.152 INFO  [445621] [CWriter::initEvent@75] CWriter::initEvent dynamic STATIC(GPU style) _record_max 221 _bounce_max  9 _steps_per_photon 10 num_g4event 1
    2019-05-30 15:17:15.152 INFO  [445621] [CRec::initEvent@87] CRec::initEvent note recstp
    2019-05-30 15:17:15.152 INFO  [445621] [CG4::propagate@330]  calling BeamOn numG4Evt 1
    2019-05-30 15:17:16.561 INFO  [445621] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2019-05-30 15:17:16.562 ERROR [445621] [GBndLib::getMaterialIndexFromLine@717]  line 7 ibnd 1 numBnd 3 imatsur 3
    2019-05-30 15:17:16.562 INFO  [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@135]  genstep_idx 0 num_gs 1 materialLine 7 materialIndex 1      post  0.000   0.000   0.000   0.000 

    2019-05-30 15:17:16.562 INFO  [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@168]  From Genstep :  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 preVelocity 276.074 postVelocity 273.253
    2019-05-30 15:17:16.562 ERROR [445621] [CCerenkovGenerator::GetRINDEX@73]  aMaterial 0x1ca5270 aMaterial.Name Water materialIndex 1 num_material 3 Rindex 0x1c27450 Rindex2 0x1c27450
    2019-05-30 15:17:16.562 ERROR [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@266]  genstep_idx 0 fNumPhotons 221 pindex 0
    2019-05-30 15:17:16.945 INFO  [445621] [CAlignEngine::CAlignEngine@75] CAlignEngine seq_index -1 seq 100000,16,16 seq_ni 100000 seq_nv 256 cur 100000 seq_path $TMP/TRngBufTest.npy simstream logpath - recycle_idx 0
    CAlignEngine seq_index -1 seq 100000,16,16 seq_ni 100000 seq_nv 256 cur 100000 seq_path $TMP/TRngBufTest.npy simstream logpath - recycle_idx 0
    (     0:   0)   0.740219 :   0x7f96a2ec4602     0x101e CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)
    2019-05-30 15:17:16.946 ERROR [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@376] gcp.u2 0.517013 dir ( 0.796318 -0.22456 0.56165 ) pol ( -0.571033 0.027155 0.820478 )
    2019-05-30 15:17:16.947 ERROR [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@401] gcp.u3 0.156989
    2019-05-30 15:17:16.947 ERROR [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@413] gcp.u4 0.0713675
    2019-05-30 15:17:16.947 ERROR [445621] [CCerenkovGenerator::GeneratePhotonsFromGenstep@444] gcp.post ( 0.053994 -0.010820 -0.002176 : 0.000204)
    (   200:   0)   0.919048 :   0x7f96a2ec4602     0x101e CCerenkovGenerator::GeneratePhotonsFromGenstep(OpticksGenstep const*, unsigned int)

     Exiting from C4Cerenkov1042::DoIt -- NumberOfSecondaries = 221
    2019-05-30 15:17:17.322 INFO  [445621] [C4PhotonCollector::collectSecondaryPhotons@70]  numberOfSecondaries 221
    2019-05-30 15:17:17.322 INFO  [445621] [CGenstepSource::addPrimaryVertices@118]  numberOfSecondaries 221
    2019-05-30 15:17:17.323 INFO  [445621] [CSensitiveDetector::Initialize@56]  HCE 0x37649f0 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    2019-05-30 15:17:17.324 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.324 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.325 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.326 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.326 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    2019-05-30 15:17:17.326 FATAL [445621] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536


Uncomment an assert.

DEBUG=1 ckm-okg4::

    Exiting from C4Cerenkov1042::DoIt -- NumberOfSecondaries = 221
    2019-05-30 15:43:20.112 INFO  [36244] [C4PhotonCollector::collectSecondaryPhotons@70]  numberOfSecondaries 221
    2019-05-30 15:43:20.112 INFO  [36244] [CGenstepSource::addPrimaryVertices@118]  numberOfSecondaries 221
    2019-05-30 15:43:20.113 INFO  [36244] [CSensitiveDetector::Initialize@56]  HCE 0x2494280 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    2019-05-30 15:43:20.114 FATAL [36244] [CPhoton::add@86] flag mismatch  _flag 1 _his 1 flag 65536
    OKG4Test: /home/blyth/opticks/cfg4/CPhoton.cc:91: void CPhoton::add(unsigned int, unsigned int): Assertion `flag_match' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe2037207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2037207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20388f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2030026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20300d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdce79f in CPhoton::add (this=0x960ed0, flag=65536, material=2) at /home/blyth/opticks/cfg4/CPhoton.cc:91
    #5  0x00007fffefdd02f6 in CWriter::writeStepPoint (this=0x90ca70, point=0x24b3130, flag=65536, material=2) at /home/blyth/opticks/cfg4/CWriter.cc:121
    #6  0x00007fffefdc76e7 in CRecorder::RecordStepPoint (this=0x960e90, point=0x24b3130, flag=65536, material=2, boundary_status=Ds::Undefined) at /home/blyth/opticks/cfg4/CRecorder.cc:468
    #7  0x00007fffefdc7064 in CRecorder::postTrackWriteSteps (this=0x960e90) at /home/blyth/opticks/cfg4/CRecorder.cc:398
    #8  0x00007fffefdc6312 in CRecorder::postTrack (this=0x960e90) at /home/blyth/opticks/cfg4/CRecorder.cc:133
    #9  0x00007fffefdedb1e in CG4::postTrack (this=0x7003d0) at /home/blyth/opticks/cfg4/CG4.cc:255
    #10 0x00007fffefde9b7e in CTrackingAction::PostUserTrackingAction (this=0x9d4580, track=0x24b1a00) at /home/blyth/opticks/cfg4/CTrackingAction.cc:91
    #11 0x00007fffec139326 in G4TrackingManager::ProcessOneTrack (this=0x88a310, apValueG4Track=0x24b1a00) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4TrackingManager.cc:140
    #12 0x00007fffec3b1d46 in G4EventManager::DoProcessing (this=0x88a280, anEvent=0x23dd290) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:185
    #13 0x00007fffec3b2572 in G4EventManager::ProcessOneEvent (this=0x88a280, anEvent=0x23dd290) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
    #14 0x00007fffec6b4665 in G4RunManager::ProcessOneEvent (this=0x706cf0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
    #15 0x00007fffec6b44d7 in G4RunManager::DoEventLoop (this=0x706cf0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #16 0x00007fffec6b3d2d in G4RunManager::BeamOn (this=0x706cf0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #17 0x00007fffefdee529 in CG4::propagate (this=0x7003d0) at /home/blyth/opticks/cfg4/CG4.cc:331
    #18 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffd770) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #19 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffd770) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #20 0x00000000004039a7 in main (argc=6, argv=0x7fffffffdaa8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 


    (gdb) f 7
#7  0x00007fffefdc7064 in CRecorder::postTrackWriteSteps (this=0x960e90) at /home/blyth/opticks/cfg4/CRecorder.cc:398
    398             done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );   
    (gdb) p preFlag
    $1 = 65536
    (gdb) p NATURAL
    $2 = NATURAL
    (gdb) p (int)NATURAL
    $3 = 65536
    (gdb) 


    (gdb) f 4
    #4  0x00007fffefdce79f in CPhoton::add (this=0x960ed0, flag=65536, material=2) at /home/blyth/opticks/cfg4/CPhoton.cc:91
    91       assert( flag_match ); 
    (gdb) l
    86         LOG(fatal) << "flag mismatch "
    87                    << " _flag " << _flag 
    88                    << " _his " << _his 
    89                    << " flag " << flag 
    90                    ; 
    91       assert( flag_match ); 
    92  
    93      _mat = material < 0xFull ? material : 0xFull ; 
    94      _material = material ; 
    95  


    (gdb) p flag
    $4 = 65536
    (gdb) p _flag 
    $5 = 1
    (gdb) p (int)CERENKOV
    $6 = 1
    (gdb) p (int)SCINTILLATION
    $7 = 2
    (gdb) 



CRecorder.cc::

    389        // as clearStp for each track, REJOIN will always be i=0
    390 
    391         unsigned preFlag = first ? m_ctx._gen : OpStatus::OpPointFlag(pre,  prior_boundary_status, stage) ;
    392 
    393         if(i == 0)
    394         {
    395 
    396             m_state._step_action |= CAction::PRE_SAVE ;
    397 
    398>            done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );
    399 
    400             if(done) m_state._step_action |= CAction::PRE_DONE ;
    401 
    402             if(!done)
    403             {
    404                  done = RecordStepPoint( post, postFlag, u_postmat, boundary_status,       POST );
    405 
    406                  m_state._step_action |= CAction::POST_SAVE ;
    407 
    408                  if(done) m_state._step_action |= CAction::POST_DONE ;





CAUSE : expecting CERENKOV but getting NATURAL 
---------------------------------------------------

:: 

    --- a/cfg4/CPhoton.cc   Thu May 30 15:08:48 2019 +0800
    +++ b/cfg4/CPhoton.cc   Thu May 30 16:55:27 2019 +0800
    @@ -77,18 +77,32 @@
         unsigned long long  msk = 0xFull << shift ; 
     
         _slot_constrained = slot ; 
    +
         _his = BBit::ffs(flag) & 0xFull ; 
     
    +    //  BBit::ffs result is a 1-based bit index of least significant set bit 
    +    //  so anding with 0xF although looking like a bug, as the result of ffs is not a nibble, 
    +    //  is actually providing a warning as are constructing seqhis from nibbles : 
    +    //  this is showing that NATURAL is too big to fit in its nibble   
    +    //
    +    //  BUT NATURAL is an input flag meaning either CERENKOV or SCINTILATION, thus
    +    //  it should not be here at the level of a photon.  It needs to be set 
    +    //  at genstep level to the appropriate thing. 
    +    //
    +    //  See notes/issues/ckm-okg4-CPhoton-add-flag-mismatch.rst
    +    //
    +
         _flag = 0x1 << (_his - 1) ; 
     
         bool flag_match = _flag == flag  ; 
         if(!flag_match)
            LOG(fatal) << "flag mismatch "
    +                  << " MAYBE TOO BIG TO FIT IN THE NIBBLE " 
                       << " _flag " << _flag 
                       << " _his " << _his 
                       << " flag " << flag 
                       ; 
    -    //assert( flag_match ); 
    +     assert( flag_match ); 



::

    In [3]: 17 & 0xF
    Out[3]: 1

    In [5]: 0x1 << (17 - 1) 
    Out[5]: 65536


::

     39 struct CFG4_API CG4Ctx
     40 {
     41     Opticks* _ok ;
     42     int   _pindex ;
     43     bool  _print ;
     44 
     45     // CG4::init
     46     bool _dbgrec ;
     47     bool _dbgseq ;
     48     bool _dbgzero ;
     49 
     50     // CG4::initEvent
     51     int  _photons_per_g4event ;
     52     unsigned  _steps_per_photon  ;
     53     unsigned  _gen  ;
     54     unsigned  _record_max ;
     55     unsigned  _bounce_max ;
     56 


::

    286 void CG4::initEvent(OpticksEvent* evt)
    287 {
    288     m_generator->configureEvent(evt);
    289 
    290     m_ctx.initEvent(evt);
    291 
    292     m_recorder->initEvent(evt);
    293 
    294     NPY<float>* nopstep = evt->getNopstepData();
    295     if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ;
    296     assert(nopstep);
    297     m_steprec->initEvent(nopstep);
    298 }

::

    139 void CG4Ctx::initEvent(const OpticksEvent* evt)
    140 {
    141     _ok_event_init = true ;
    142     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    143     _steps_per_photon = evt->getMaxRec() ;
    144     _record_max = evt->getNumPhotons();   // from the genstep summation
    145     _bounce_max = evt->getBounceMax();
    146 
    147     const char* typ = evt->getTyp();
    148     _gen = OpticksFlags::SourceCode(typ);
    149 
    150     LOG(info) << "CG4Ctx::initEvent"
    151               << " _record_max (numPhotons from genstep summation) " << _record_max
    152               << " photons_per_g4event " << _photons_per_g4event
    153               << " steps_per_photon " << _steps_per_photon
    154               << " typ " << typ
    155               << " gen " << _gen
    156               << " SourceType " << OpticksFlags::SourceType(_gen)
    157               ;
    158 
    159     assert( _gen == TORCH || _gen == G4GUN || _gen == NATURAL );  // what is needed to add NATURAL ?
    160 }


* should not be natural instead : CERENKOV || SCINTILLATION 



Perhaps just switch FABRICATED and NATURAL ? 16 is also too large too (ffs is 1-based) and do not want to have zero 
(an empty nibble) meaning something other than empty::

    blyth@localhost optickscore]$ cat OpticksPhoton.h
    #pragma once

    enum
    {
        CERENKOV          = 0x1 <<  0,    
        SCINTILLATION     = 0x1 <<  1,    
        MISS              = 0x1 <<  2,
        BULK_ABSORB       = 0x1 <<  3,
        BULK_REEMIT       = 0x1 <<  4,
        BULK_SCATTER      = 0x1 <<  5,
        SURFACE_DETECT    = 0x1 <<  6,
        SURFACE_ABSORB    = 0x1 <<  7,
        SURFACE_DREFLECT  = 0x1 <<  8,
        SURFACE_SREFLECT  = 0x1 <<  9,
        BOUNDARY_REFLECT  = 0x1 << 10,
        BOUNDARY_TRANSMIT = 0x1 << 11,
        TORCH             = 0x1 << 12,
        NAN_ABORT         = 0x1 << 13,
        G4GUN             = 0x1 << 14,
        FABRICATED        = 0x1 << 15,
        NATURAL           = 0x1 << 16,
        MACHINERY         = 0x1 << 17,
        EMITSOURCE        = 0x1 << 18,
        PRIMARYSOURCE     = 0x1 << 19,
        GENSTEPSOURCE     = 0x1 << 20
    }; 

    //  only ffs 0-15 make it into the record so debug flags only beyond 15 





Multiple genstep of different types per event ?
----------------------------------------------------

* hmm multiple types of genstep per OpticksEvent ?

  * the types of genstep per G4Event is the relevant thing, as CFG4 is all about 
    recording the Geant4 propagation following in the exact same format as Opticks GPU propagation does 


::

     36 CGenstepSource::CGenstepSource(Opticks* ok, NPY<float>* gs)
     37     :  
     38     CSource(ok),
     39     m_gs(new OpticksGenstep(gs)),
     40     m_num_genstep(m_gs->getNumGensteps()),
     41     m_num_genstep_per_g4event(1),
     42     m_tranche(new STranche(m_num_genstep,m_num_genstep_per_g4event)),
     43     m_idx(0),
     44     m_generate_count(0),
     45     m_photon_collector(new C4PhotonCollector)
     46 {   
     47     init();
     48 }      




