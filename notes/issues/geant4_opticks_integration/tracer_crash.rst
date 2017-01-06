Tracer Crash
==============

Still get the crash with OTracerTest unless use `--load` option
despite loading an event not making sense when just tracing::

    OTracerTest --load

    // TODO: tidy this 


Tidy Actions
--------------

* add argforced "--tracer" option to OTracerTest and isTracer to Opticks
* use the isTracer to skip propagator setup in okop-/OpEngine






Huh, why is genstep_buffer causing a tracing issue ?

Perhsaps resolved with::

    --- a/okop/OpEngine.cc  Wed Nov 23 20:04:06 2016 +0800
    +++ b/okop/OpEngine.cc  Fri Nov 25 19:43:16 2016 +0800
    @@ -43,17 +43,40 @@
           m_ok(m_hub->getOpticks()),
           m_scene(new OScene(m_hub)),
           m_ocontext(m_scene->getOContext()),
    -      m_entry(m_ocontext->addEntry(m_ok->getEntryCode())),
    -      m_oevt(new OEvent(m_hub, m_ocontext)),
    -      m_propagator(new OPropagator(m_hub, m_oevt, m_entry)),
    -      m_seeder(new OpSeeder(m_hub, m_oevt)),
    -      m_zeroer(new OpZeroer(m_hub, m_oevt)),
    -      m_indexer(new OpIndexer(m_hub, m_oevt))
    +      m_entry(NULL),
    +      m_oevt(NULL),
    +      m_propagator(NULL),
    +      m_seeder(NULL),
    +      m_zeroer(NULL),
    +      m_indexer(NULL)
    +{
    +   init();
    +   (*m_log)("DONE");
    +}
    +void OpEngine::init()
     {
        m_ok->setOptiXVersion(OConfig::OptiXVersion()); 
    -   (*m_log)("DONE");
    +   if(m_ok->isLoad())
    +   {
    +       LOG(warning) << "OpEngine::init skip initPropagation as just loading pre-cooked event " ;
    +   }
    +   else
    +   {
    +       initPropagation(); 
    +   }
     }
     
    +void OpEngine::initPropagation()
    +{
    +    m_entry = m_ocontext->addEntry(m_ok->getEntryCode()) ;
    +    m_oevt = new OEvent(m_hub, m_ocontext);
    +    m_propagator = new OPropagator(m_hub, m_oevt, m_entry);
    +    m_seeder = new OpSeeder(m_hub, m_oevt) ;
    +    m_zeroer = new OpZeroer(m_hub, m_oevt) ;
    +    m_indexer = new OpIndexer(m_hub, m_oevt) ;
    +}
    +




::

    Interactor::key_pressed 340 
    2016-10-31 17:56:05.742 INFO  [391393] [Interactor::key_pressed@428] Interactor::key_pressed O nextRenderStyle 
    2016-10-31 17:56:05.742 INFO  [391393] [OTracer::trace_@128] OTracer::trace  entry_index 1 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.00975,-142.111) front 0.6618,0.7442,-0.0906
    2016-10-31 17:56:05.742 INFO  [391393] [OContext::close@216] OContext::close numEntryPoint 2
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Initalization of non-primitive type genstep_buffer:  Buffer object, [1769674])
    Process 16111 stopped
    * thread #1: tid = 0x5f8e1, 0x00007fff91a2c866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff91a2c866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff91a2c866:  jae    0x7fff91a2c870            ; __pthread_kill + 20
       0x7fff91a2c868:  movq   %rax, %rdi
       0x7fff91a2c86b:  jmp    0x7fff91a29175            ; cerror_nocancel
       0x7fff91a2c870:  retq   
    (lldb) bt
    * thread #1: tid = 0x5f8e1, 0x00007fff91a2c866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff91a2c866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff890c935c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8fe19b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8f6d9f31 libc++abi.dylib`abort_message + 257
        frame #4: 0x00007fff8f6ff93a libc++abi.dylib`default_terminate_handler() + 240
        frame #5: 0x00007fff8fa37322 libobjc.A.dylib`_objc_terminate() + 124
        frame #6: 0x00007fff8f6fd1d1 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff8f6fcc5b libc++abi.dylib`__cxa_throw + 124
        frame #8: 0x000000010310b0c9 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x0000000119388860, code=RT_ERROR_INVALID_VALUE) const + 121 at optixpp_namespace.h:1840
        frame #9: 0x000000010311e187 libOptiXRap.dylib`optix::ContextObj::compile(this=0x0000000119388860) + 55 at optixpp_namespace.h:2376
        frame #10: 0x000000010311d834 libOptiXRap.dylib`OContext::launch(this=0x0000000119388a10, lmode=30, entry=1, width=2880, height=1704, times=0x000000010d83b7a0) + 660 at OContext.cc:268
        frame #11: 0x000000010312eccb libOptiXRap.dylib`OTracer::trace_(this=0x000000010d83e320) + 3995 at OTracer.cc:142
        frame #12: 0x000000010377ba4b libOpticksGL.dylib`OKGLTracer::render(this=0x000000010d839520) + 123 at OKGLTracer.cc:109
        frame #13: 0x000000010202fbd4 libOGLRap.dylib`OpticksViz::render(this=0x000000010c1fbca0) + 132 at OpticksViz.cc:401
        frame #14: 0x000000010202ee9a libOGLRap.dylib`OpticksViz::renderLoop(this=0x000000010c1fbca0) + 906 at OpticksViz.cc:443
        frame #15: 0x000000010202e5f2 libOGLRap.dylib`OpticksViz::visualize(this=0x000000010c1fbca0) + 34 at OpticksViz.cc:129
        frame #16: 0x0000000103fac63f libokg4.dylib`OKG4Mgr::visualize(this=0x00007fff5fbfe560) + 47 at OKG4Mgr.cc:110
        frame #17: 0x00000001000139db OKG4Test`main(argc=23, argv=0x00007fff5fbfe640) + 1515 at OKG4Test.cc:58
        frame #18: 0x00007fff8ce9f5fd libdyld.dylib`start + 1



OpEngine ctor is adding entry G for generate::

    (lldb) f 5
    frame #5: 0x00000001036b3207 libOKOP.dylib`OpEngine::OpEngine(this=0x000000010c038260, hub=0x000000010560ada0) + 247 at OpEngine.cc:46
       43         m_ok(m_hub->getOpticks()),
       44         m_scene(new OScene(m_hub)),
       45         m_ocontext(m_scene->getOContext()),
    -> 46         m_entry(m_ocontext->addEntry(m_ok->getEntryCode())),
       47         m_oevt(new OEvent(m_hub, m_ocontext)),
       48         m_propagator(new OPropagator(m_hub, m_oevt, m_entry)),
       49         m_seeder(new OpSeeder(m_hub, m_oevt)),
    (lldb) f 4
    frame #4: 0x000000010311374d libOptiXRap.dylib`OContext::addEntry(this=0x0000000116f41c70, code='G') + 285 at OContext.cc:45
       42   OpticksEntry* OContext::addEntry(char code)
       43   {
       44       LOG(fatal) << "OContext::addEntry " << code ; 
    -> 45       assert(0);
       46       bool defer = true ; 
       47       unsigned index ;
       48       switch(code)
    (lldb) p code
    (char) $0 = 'G'
    (lldb) 


    (lldb) f 7
    frame #7: 0x00000001037a1a94 libOK.dylib`OKPropagator::OKPropagator(this=0x000000010c038200, hub=0x000000010560ada0, idx=0x00000001087f8740, viz=0x00000001087f8e50) + 196 at OKPropagator.cc:44
       41       m_hub(hub),    
       42       m_idx(idx),
       43       m_viz(viz),    
    -> 44       m_ok(m_hub->getOpticks()),
       45   #ifdef WITH_OPTIX
       46       m_engine(new OpEngine(m_hub)),
       47       m_tracer(m_viz ? new OKGLTracer(m_engine,m_viz, true) : NULL ),
    (lldb) 



