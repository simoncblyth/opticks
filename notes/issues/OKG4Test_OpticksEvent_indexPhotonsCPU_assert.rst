FIXED : OKG4Test_OpticksEvent_indexPhotonsCPU_assert
=======================================================

Review CRecorder/CWriter : its writing photons, records and history 
in either dynamic or static mode wrt knowing the number of photons ahead of time.

Added `--primarysource` and `--emitsource` as is currently defaulting to `--torch` which is not-dynamic
whereas `--primarysource` needs to be dynamic.

But fixed the issue by directly recording the dynamic nature in OpticksEvent and using 
that to control the resize. 


::

    2018-08-20 14:27:38.030 INFO  [99120] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2018-08-20 14:27:38.030 INFO  [99120] [CG4::postpropagate@335] CG4::postpropagate(0) ctx CG4Ctx::desc_stats dump_count 0 event_total 1 event_track_count 87
    2018-08-20 14:27:38.030 INFO  [99120] [OpticksEvent::postPropagateGeant4@2049] OpticksEvent::postPropagateGeant4 shape  genstep NULL nopstep 94,4,4 photon 54,4,4 source NULL record 54,10,2,4 phosel 0,1,4 recsel 0,10,1,4 sequence 54,1,2 seed 0,1,1 hit 0,4,4 num_photons 54
    2018-08-20 14:27:38.030 FATAL [99120] [OpticksEvent::postPropagateGeant4@2067]  NOT setting num_photons 54
    2018-08-20 14:27:38.030 INFO  [99120] [OpticksEvent::indexPhotonsCPU@2117] OpticksEvent::indexPhotonsCPU sequence 54,1,2 phosel 0,1,4 phosel.hasData 0 recsel0 0,10,1,4 recsel0.hasData 0
    2018-08-20 14:27:38.030 FATAL [99120] [OpticksEvent::indexPhotonsCPU@2137]  length mismatch  sequence : 54 phosel   : 0
    Assertion failed: (sequence->getShape(0) == phosel->getShape(0)), function indexPhotonsCPU, file /Users/blyth/opticks/optickscore/OpticksEvent.cc, line 2146.
    Process 9413 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7560bb6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff7560bb6e <+10>: jae    0x7fff7560bb78            ; <+20>
        0x7fff7560bb70 <+12>: movq   %rax, %rdi
        0x7fff7560bb73 <+15>: jmp    0x7fff75602b00            ; cerror_nocancel
        0x7fff7560bb78 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff7560bb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff757d6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff755671ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7552f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010de686e0 libOpticksCore.dylib`OpticksEvent::indexPhotonsCPU(this=0x00000001121b7150) at OpticksEvent.cc:2146
        frame #5: 0x000000010de67fff libOpticksCore.dylib`OpticksEvent::postPropagateGeant4(this=0x00000001121b7150) at OpticksEvent.cc:2071
        frame #6: 0x0000000106a585d5 libCFG4.dylib`CG4::postpropagate(this=0x0000000110fa4d70) at CG4.cc:344
        frame #7: 0x0000000106a580b8 libCFG4.dylib`CG4::propagate(this=0x0000000110fa4d70) at CG4.cc:323
        frame #8: 0x00000001000df2b6 libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe918) at OKG4Mgr.cc:141
        frame #9: 0x00000001000deec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe918) at OKG4Mgr.cc:84
        frame #10: 0x00000001000148b9 OKG4Test`main(argc=5, argv=0x00007ffeefbfea08) at OKG4Test.cc:9
        frame #11: 0x00007fff754bb015 libdyld.dylib`start + 1
        frame #12: 0x00007fff754bb015 libdyld.dylib`start + 1
    (lldb) 


