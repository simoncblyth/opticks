ckm-okg4-natural-fail
========================

::

   ckm-okg4(){      OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save --natural ;}


Added natural to *ckm-okg4* deal in the same gensteps as the others, but OKG4Test is 
not expecting "genstep" source running::

    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksEvent::setBufferControl@963]    genstep : (spec) : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE VERBOSE_MODE  : Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1 20190313_193117 OKG4Test
    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksRun::passBaton@170] OpticksRun::passBaton nopstep 0x1119b5020 genstep 0x111904990 source 0x0
    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksEvent::setBufferControl@963]    genstep : (spec) : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE VERBOSE_MODE  : Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/1 20190313_193117 OKG4Test
    2019-03-13 19:31:17.636 INFO  [1512502] [*CG4::propagate@304] Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1 20190313_193117 OKG4Test  genstep 1,6,4 nopstep 0,4,4 photon 221,4,4 source NULL record 221,10,2,4 phosel 221,1,4 recsel 221,10,1,4 sequence 221,1,2 seed 221,1,1 hit 0,4,4
    2019-03-13 19:31:17.636 INFO  [1512502] [*CG4::propagate@322] CG4::propagate(0) /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1
    2019-03-13 19:31:17.636 INFO  [1512502] [CGenerator::configureEvent@104] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    Assertion failed: (_gen == TORCH || _gen == G4GUN), function initEvent, file /Users/blyth/opticks/cfg4/CG4Ctx.cc, line 149.
    Process 25455 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff65123b66 <+10>: jae    0x7fff65123b70            ; <+20>
        0x7fff65123b68 <+12>: movq   %rax, %rdi
        0x7fff65123b6b <+15>: jmp    0x7fff6511aae9            ; cerror_nocancel
        0x7fff65123b70 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff652ee080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff6507f1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff650471ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106a63362 libCFG4.dylib`CG4Ctx::initEvent(this=0x0000000111904ee8, evt=0x00000001119941c0) at CG4Ctx.cc:149
        frame #5: 0x0000000106a682b8 libCFG4.dylib`CG4::initEvent(this=0x0000000111904ec0, evt=0x00000001119941c0) at CG4.cc:290
        frame #6: 0x0000000106a68ae7 libCFG4.dylib`CG4::propagate(this=0x0000000111904ec0) at CG4.cc:324
        frame #7: 0x00000001000e229a libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe7d8) at OKG4Mgr.cc:137
        frame #8: 0x00000001000e1ec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe7d8) at OKG4Mgr.cc:84
        frame #9: 0x0000000100014c7e OKG4Test`main(argc=6, argv=0x00007ffeefbfe8a8) at OKG4Test.cc:9
        frame #10: 0x00007fff64fd3015 libdyld.dylib`start + 1
        frame #11: 0x00007fff64fd3015 libdyld.dylib`start + 1
    (lldb) 

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
    149     assert( _gen == TORCH || _gen == G4GUN  );
    150 
    151     LOG(info) << "CG4Ctx::initEvent"
    152               << " _record_max (numPhotons from genstep summation) " << _record_max
    153               << " photons_per_g4event " << _photons_per_g4event
    154               << " steps_per_photon " << _steps_per_photon
    155               << " gen " << _gen
    156               ;
    157 }

