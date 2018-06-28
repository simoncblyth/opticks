oevent-downloadhits-crash
===================================


TODO:

* abort on receipt of unknown option arguments like "--cfg4" 


Which executable for aligned running ? Must use OKG4Mgr, eg OKG4Test 
------------------------------------------------------------------------

::

    099 void OKG4Mgr::propagate_()
    100 {
    101     // Normally the G4 propagation is done first, because 
    102     // gensteps eg from G4Gun can then be passed to Opticks.
    103     // However with RNG-aligned testing using "--align" option
    104     // which uses emitconfig CPU generated photons there is 
    105     // no need to do G4 first. Actually it is more convenient
    106     // for Opticks to go first in order to allow access to the ucf.py 
    107     // parsed  kernel pindex log during lldb python scripted G4 debugging.
    108 
    109     bool align = m_ok->isAlign();
    110 
    111     if(m_ok->isFabricatedGensteps())  // eg torch running 
    112     {
    113          NPY<float>* gs = m_gen->getInputGensteps() ;
    114 
    115          m_run->setGensteps(gs);
    116 
    117          if(align)
    118              m_propagator->propagate();
    119 
    120 
    121          m_g4->propagate();
    122     }
    123     else
    124     {
    125          NPY<float>* gs = m_g4->propagate() ;
    126 
    127          if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ;
    128          assert(gs);
    129 
    130          m_run->setGensteps(gs);
    131     }
    132 
    133     if(!align)
    134         m_propagator->propagate();
    135 }




::

    169 op-binary-name-default(){ echo OKTest ; }
    170 
    171 op-binary-names(){ type op-binary-name | perl -ne 'm,--(\w*)\), && print "$1\n" ' - ; }
    172 op-binary-name()
    173 {
    174    case $1 in
    175          --version) echo OpticksCMakeConfigTest ;;
    176          --idpath) echo OpticksIDPATH ;;
    177            --keys) echo InteractorKeys ;;
    178           --tcfg4) echo CG4Test ;;
    179            --okg4) echo OKG4Test ;;
    180          --tracer) echo OTracerTest ;;
    181       --gdml2gltf) echo gdml2gltf.py ;;
    182             --mat) echo GMaterialLibTest ;;
    183            --cmat) echo CMaterialLibTest ;;
    184            --surf) echo GSurfaceLibTest ;;
    185            --snap) echo OpTest ;;




Was trying to follow along some :doc:`random_alignment_iterating` commands  



::

    epsilon:optickscore blyth$ tboolean-;tboolean-box --cfg4 -D        // THIS GIVE THE BELOW ERROR
    epsilon:optickscore blyth$ tboolean-;tboolean-box --tcfg4 -D       // THIS WORKS

The wrong option resulted in the wrong executable OKTest being used rather than CG4Test.

::

    2018-06-27 11:22:49.925 INFO  [27807337] [OPropagator::prelaunch@166] 1 : (0;100000,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.000011.3426 0.0000
    2018-06-27 11:22:49.925 ERROR [27807337] [OPropagator::launch@186] LAUNCH NOW -
    2018-06-27 11:22:50.260 ERROR [27807337] [OPropagator::launch@188] LAUNCH DONE
    2018-06-27 11:22:50.260 INFO  [27807337] [OPropagator::launch@191] 1 : (0;100000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.3351
    2018-06-27 11:22:50.260 INFO  [27807337] [OpIndexer::indexSequenceInterop@260] OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq 
    2018-06-27 11:22:50.337 INFO  [27807337] [OpticksViz::indexPresentationPrep@325] OpticksViz::indexPresentationPrep
    2018-06-27 11:22:50.338 ERROR [27807337] [*GBndLib::createBufferForTex2d@697] GBndLib::createBufferForTex2d mat 0x10ee1d0f0 sur 0x10ee1d2f0
    2018-06-27 11:22:50.338 INFO  [27807337] [GPropertyLib::close@418] GPropertyLib::close type GBndLib buf 2,4,2,39,4
    2018-06-27 11:22:50.339 INFO  [27807337] [OpticksViz::downloadEvent@315] OpticksViz::downloadEvent (1)
    2018-06-27 11:22:50.444 INFO  [27807337] [Rdr::download@72] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2018-06-27 11:22:50.444 INFO  [27807337] [OpticksViz::downloadEvent@317] OpticksViz::downloadEvent (1) DONE 
    2018-06-27 11:22:50.444 INFO  [27807337] [OEvent::download@352] OEvent::download id 1
    2018-06-27 11:22:50.444 INFO  [27807337] [OContext::download@434] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void **)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    Process 70862 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (OKTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff561cc080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff55f5d1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff53e61f8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff53e62113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff55299eab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff53e7d7c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff53e7d26f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x0000000100480e85 libOptiXRap.dylib`optix::APIObj::checkError(this=0x000000011dbb7c90, code=RT_ERROR_INVALID_VALUE) const at optixpp_namespace.h:1936
        frame #9: 0x00000001004d228a libOptiXRap.dylib`OBufBase::bufspec() + 58
        frame #10: 0x00000001004b0814 libOptiXRap.dylib`OEvent::downloadHits(this=0x000000011ddbbbb0, evt=0x000000011da1a1d0) at OEvent.cc:393
        frame #11: 0x00000001004b09ad libOptiXRap.dylib`OEvent::download(this=0x000000011ddbbbb0) at OEvent.cc:342
        frame #12: 0x00000001003d0649 libOKOP.dylib`OpEngine::downloadEvent(this=0x000000010f033ee0) at OpEngine.cc:122
        frame #13: 0x00000001000d3dc0 libOK.dylib`OKPropagator::downloadEvent(this=0x000000010f033ea0) at OKPropagator.cc:108
        frame #14: 0x00000001000d3a68 libOK.dylib`OKPropagator::propagate(this=0x000000010f033ea0) at OKPropagator.cc:82
        frame #15: 0x00000001000d3437 libOK.dylib`OKMgr::propagate(this=0x00007ffeefbfdda8) at OKMgr.cc:102
        frame #16: 0x000000010000b9a1 OKTest`main(argc=29, argv=0x00007ffeefbfde88) at OKTest.cc:14
        frame #17: 0x00007fff55eb1015 libdyld.dylib`start + 1
        frame #18: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) 


::


    2018-06-27 11:31:36.657 INFO  [27814494] [OpticksViz::indexPresentationPrep@325] OpticksViz::indexPresentationPrep
    2018-06-27 11:31:36.658 ERROR [27814494] [*GBndLib::createBufferForTex2d@697] GBndLib::createBufferForTex2d mat 0x11249bc50 sur 0x10e500450
    2018-06-27 11:31:36.658 INFO  [27814494] [GPropertyLib::close@418] GPropertyLib::close type GBndLib buf 3,4,2,39,4
    2018-06-27 11:31:36.658 INFO  [27814494] [OpticksViz::downloadEvent@315] OpticksViz::downloadEvent (1)
    2018-06-27 11:31:36.759 INFO  [27814494] [Rdr::download@72] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2018-06-27 11:31:36.759 INFO  [27814494] [OpticksViz::downloadEvent@317] OpticksViz::downloadEvent (1) DONE 
    2018-06-27 11:31:36.759 INFO  [27814494] [OEvent::download@352] OEvent::download id 1
    2018-06-27 11:31:36.759 INFO  [27814494] [OContext::download@434] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void **)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    Process 72061 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) 



::

    2018-06-27 11:36:08.222 INFO  [27817968] [Rdr::download@72] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2018-06-27 11:36:08.222 INFO  [27817968] [OpticksViz::downloadEvent@317] OpticksViz::downloadEvent (1) DONE 
    2018-06-27 11:36:08.222 INFO  [27817968] [OEvent::download@352] OEvent::download id 1
    2018-06-27 11:36:08.222 INFO  [27817968] [OContext::download@434] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void **)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    Process 72614 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) 




