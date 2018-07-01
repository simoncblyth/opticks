FIXED : tboolean-interlocked
===============================

Why is the generate.cu even in the OptiX context in tracer mode ? Should
be only the pinhole camera ?

There was a problem with options getting thru, due to Opticks::GetInstance in OKMgr
causing instanciation of a 2nd Opticks. 


::

    epsilon:optixrap blyth$ tboolean-;tboolean-interlocked --tracer --nopropagate -D

    2018-07-01 16:26:16.746 INFO  [2135824] [SLog::operator@20] OKPropagator::OKPropagator  DONE
    OKMgr::init
       OptiXVersion :           50001
    2018-07-01 16:26:16.746 INFO  [2135824] [SLog::operator@20] OKMgr::OKMgr  DONE
    2018-07-01 16:26:16.746 INFO  [2135824] [Bookmarks::create@249] Bookmarks::create : persisting state to slot 0
    2018-07-01 16:26:16.747 INFO  [2135824] [Bookmarks::collect@273] Bookmarks::collect 0
    2018-07-01 16:26:16.749 WARN  [2135824] [OpticksViz::prepareGUI@368] App::prepareGUI NULL TimesTable 
    2018-07-01 16:26:16.749 INFO  [2135824] [OpticksViz::renderLoop@449] enter runloop 
    2018-07-01 16:26:16.752 INFO  [2135824] [OpticksViz::renderLoop@454] after frame.show() 
    2018-07-01 16:26:16.951 INFO  [2135824] [Animator::Summary@313] Composition::gui setup Animation   OFF 0/0/    0.0000
    2018-07-01 16:26:16.951 INFO  [2135824] [Animator::Summary@313] Composition::initRotator   OFF 0/0/    0.0000
    2018-07-01 16:26:18.291 INFO  [2135824] [Interactor::key_pressed@409] Interactor::key_pressed O nextRenderStyle 
    2018-07-01 16:26:18.292 INFO  [2135824] [OTracer::trace_@128] OTracer::trace  entry_index 1 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04082,-204.082) front -1.0000,0.0000,0.0000
    2018-07-01 16:26:18.292 INFO  [2135824] [OContext::close@236] OContext::close numEntryPoint 2
    2018-07-01 16:26:18.292 INFO  [2135824] [OContext::close@240] OContext::close setEntryPointCount done.
    2018-07-01 16:26:18.498 INFO  [2135824] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable record_buffer from _Z8generatev_cp5" not found in scope)
    Process 4456 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff734e6b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff734e6b6e <+10>: jae    0x7fff734e6b78            ; <+20>
        0x7fff734e6b70 <+12>: movq   %rax, %rdi
        0x7fff734e6b73 <+15>: jmp    0x7fff734ddb00            ; cerror_nocancel
        0x7fff734e6b78 <+20>: retq   
    Target 0: (OTracerTest) stopped.
    (lldb) exit
    Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] 
    /Users/blyth/opticks/bin/op.sh RC 0
    epsilon:optixrap blyth$ 
    epsilon:optixrap blyth$ 

