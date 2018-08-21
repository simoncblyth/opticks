2018-08-21-opticks-t-fails-4-of-359  FIXED 3 : NOW 1/359 : 1 KNOWN ISSUE 
=========================================================================


totals 4 / 359 
----------------------

::


    .  19 /24  Test #19 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    1.12   
              ## known issue, in default geocache : nofix : as expect will go away when update default geocache 

      1  /1   Test #1  : G4OKTest.G4OKTest                             ***Exception: SegFault         0.08   
              ## trivial issue fixed


    FAILS:
      7  /24  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.90   
              ## FIXED : inconsistency of dynamic or not and having gensteps 

      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    18.01  
              ## gensteps assert tripped at upload event


FIXED 3 : DOWN TO ONE KNOWN ISSUE 
-----------------------------------

::

    CTestLog :                 okg4 :      0/     1 : 2018-08-21 14:08:31.037943 : /usr/local/opticks/build/okg4/ctest.log 
    CTestLog :                 g4ok :      0/     1 : 2018-08-21 14:08:31.247570 : /usr/local/opticks/build/g4ok/ctest.log 
     totals  1   / 359 


    FAILS:
      19 /24  Test #19 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    1.06   
    epsilon:build blyth$ 


           


CG4Test : FIXED dynamic/static confusion bug : twas from CGenerator initializer list ordering : m_source must go last 
-----------------------------------------------------------------------------------------------------------------------

::


    epsilon:build blyth$ lldb CG4Test 
    ...
    2018-08-21 10:52:21.294 INFO  [500293] [CG4Ctx::initEvent@151] CG4Ctx::initEvent photons_per_g4event 0 steps_per_photon 10 gen 4096
    2018-08-21 10:52:21.294 INFO  [500293] [CWriter::initEvent@79] CWriter::initEvent dynamic DYNAMIC(CPU style) record_max 100000 bounce_max  9 steps_per_photon 10 num_g4event 1
    Assertion failed: (m_ctx._record_max == 0), function initEvent, file /Users/blyth/opticks/cfg4/CWriter.cc, line 89.
    ...
        frame #3: 0x00007fff7552f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001c6d0a libCFG4.dylib`CWriter::initEvent(this=0x000000010fc08370, evt=0x000000010fc51070) at CWriter.cc:89
        frame #5: 0x00000001001b35b7 libCFG4.dylib`CRecorder::initEvent(this=0x000000010fc081c0, evt=0x000000010fc51070) at CRecorder.cc:91
        frame #6: 0x00000001001ef99f libCFG4.dylib`CG4::initEvent(this=0x000000010e832390, evt=0x000000010fc51070) at CG4.cc:281
        frame #7: 0x00000001001f01b7 libCFG4.dylib`CG4::propagate(this=0x000000010e832390) at CG4.cc:313
        frame #8: 0x000000010000fa3f CG4Test`main(argc=1, argv=0x00007ffeefbfeab8) at CG4Test.cc:56
        frame #9: 0x00007fff754bb015 libdyld.dylib`start + 1
        frame #10: 0x00007fff754bb015 libdyld.dylib`start + 1
    (lldb) 
    (lldb) f 5
    frame #5: 0x00000001001b35b7 libCFG4.dylib`CRecorder::initEvent(this=0x000000010fc081c0, evt=0x000000010fc51070) at CRecorder.cc:91
       88  	void CRecorder::initEvent(OpticksEvent* evt)  // called by CG4::initEvent
       89  	{
       90  	    assert(evt);
    -> 91  	    m_writer->initEvent(evt);
       92  	    m_crec->initEvent(evt);
       93  	}
       94  	
    (lldb) 
    (lldb) f 4
    frame #4: 0x00000001001c6d0a libCFG4.dylib`CWriter::initEvent(this=0x000000010fc08370, evt=0x000000010fc51070) at CWriter.cc:89
       86  	
       87  	    if(m_dynamic)
       88  	    {
    -> 89  	        assert(m_ctx._record_max == 0 );
       90  	
       91  	        // shapes must match OpticksEvent::createBuffers
       92  	        // TODO: avoid this duplicity using the spec
    (lldb) 


The _record_max comes from genstep summation, so in dynamic mode
you dont have gensteps ahead of time hence zero is expected. So why is 
an event with gensteps already being set as dynamic ? 

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



Currently CWriter ctor argument dictates the dynamic or not nature
of the event.  Probably better for dynamic property to be set into 
the event : and the writer then uses that.

::

     34 CWriter::CWriter(CG4* g4, CPhoton& photon, bool dynamic)
     35    :
     36    m_g4(g4),  
     37    m_photon(photon),
     38    m_dynamic(dynamic),
     39    m_ctx(g4->getCtx()),
     40    m_ok(g4->getOpticks()),
     41    m_enabled(true),
     42 
     43    m_evt(NULL),
     44 
     45    m_primary(NULL),
     46     
     47    m_records_buffer(NULL),
     48    m_photons_buffer(NULL),
     49    m_history_buffer(NULL),
     50        
     51    m_dynamic_records(NULL),
     52    m_dynamic_photons(NULL),
     53    m_dynamic_history(NULL)
     54 {
     55 }
     56 
     57 void CWriter::setEnabled(bool enabled)
     58 {
     59     m_enabled = enabled ;
     60 }
     61 
     62 /**
     63 CWriter::initEvent
     64 -------------------
     65 
     66 Gets refs to the history, photons and records buffers from the event.
     67 When dynamic the records target is single item dynamic_records otherwise
     68 goes direct to the records_buffer.
     69 
     70 **/
     71 
     72 void CWriter::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
     73 {
     74     m_evt = evt ;
     75     assert(m_evt && m_evt->isG4());
     76 
     77     m_evt->setDynamic( m_dynamic ? 1 : 0 ) ;


::

     56 CRecorder::CRecorder(CG4* g4, CGeometry* geometry, bool dynamic)
     57    :
     58    m_g4(g4),
     59    m_ctx(g4->getCtx()),
     60    m_ok(g4->getOpticks()),
     61    m_recpoi(m_ok->isRecPoi()),
     62    m_reccf(m_ok->isRecCf()),
     63    m_state(m_ctx),
     64    m_photon(m_ctx, m_state),
     65 
     66    m_crec(new CRec(m_g4, m_state)),
     67    m_dbg(m_ctx.is_dbg() ? new CDebug(g4, m_photon, this) : NULL),
     68 
     69    m_evt(NULL),
     70    m_geometry(geometry),
     71    m_material_bridge(NULL),
     72    m_dynamic(dynamic),
     73    m_live(false),
     74    m_writer(new CWriter(g4, m_photon, m_dynamic)),
     75    m_not_done_count(0)
     76 {  

Dynamic coming from CGenerator::

    103 CG4::CG4(OpticksHub* hub)
    104    :
    105      m_hub(hub),
    106      m_ok(m_hub->getOpticks()),
    107      m_run(m_ok->getRun()),
    108      m_cfg(m_ok->getCfg()),
    109      m_ctx(m_ok),
    110      m_engine(m_ok->isAlign() ? new CRandomEngine(this) : NULL ),
    111      m_physics(new CPhysics(this)),
    112      m_runManager(m_physics->getRunManager()),
    113      m_sd(new CSensitiveDetector("SD0")),
    114      m_geometry(new CGeometry(m_hub, m_sd)),
    115      m_hookup(m_geometry->hookup(this)),
    116      m_mlib(m_geometry->getMaterialLib()),
    117      m_detector(m_geometry->getDetector()),
    118      m_generator(new CGenerator(m_hub->getGen(), this)),
    119      m_dynamic(m_generator->isDynamic()),
    120      m_collector(NULL),   // deferred instanciation until CG4::postinitialize after G4 materials have overridden lookupA
    121      m_primary_collector(new CPrimaryCollector),
    122      m_recorder(new CRecorder(this, m_geometry, m_dynamic)),
    123      m_steprec(new CStepRec(m_ok, m_dynamic)),
    124      m_visManager(NULL),
    125      m_uiManager(NULL),
    126      m_ui(NULL),
    127      m_pga(new CPrimaryGeneratorAction(m_generator->getSource())),
    128      m_sa(new CSteppingAction(this, m_generator->isDynamic())),
    129      m_ta(new CTrackingAction(this)),
    130      m_ra(new CRunAction(m_hub)),
    131      m_ea(new CEventAction(this)),
    132      m_rt(new CRayTracer(this)),
    133      m_initialized(false)
    134 {

TOO MANY INDEPENDENT m_dynamic. Makes this fragile.
Can this be cleaned up by getting rid of all apart from the one inside OpticksEvent ?

* not really because the event come an go... so treat the one in m_generator 
  and get it from there 


Dynamic gets set by the CGenerator init-ing the source::

     46 CSource* CGenerator::initSource(unsigned code)
     47 {   
     48     const char* sourceType = OpticksFlags::SourceType(code);
     49     
     50     LOG(info) << "CGenerator::makeSource"
     51               << " code " << code
     52               << " type " << sourceType
     53               ;
     54     
     55     CSource* source = NULL ;
     56     
     57     if(     code == G4GUN)      source = initG4GunSource();
     58     else if(code == TORCH)      source = initTorchSource();
     59     else if(code == EMITSOURCE) source = initInputPhotonSource();
     60     else if(code == PRIMARYSOURCE) source = initInputPrimarySource();
     61     
     62     assert(source) ;
     63     
     64     return source ;
     65 }

Maybe the place to set dynamic into the source is here::

    082 /**
     83 CGenerator::configureEvent
     84 ---------------------------
     85 
     86 Invoked from CG4::initEvent/CG4::propagate record 
     87 generator config into the OpticksEvent.
     88 
     89 **/
     90 
     91 void CGenerator::configureEvent(OpticksEvent* evt)
     92 {  
     93    if(hasGensteps())
     94    {
     95         LOG(info) << "CGenerator:configureEvent"
     96                   << " fabricated TORCH genstep (STATIC RUNNING) "
     97                   ;
     98 
     99         evt->setNumG4Event(getNumG4Event());
    100         evt->setNumPhotonsPerG4Event(getNumPhotonsPerG4Event()) ;
    101         evt->zero();  // static approach requires allocation ahead
    102 
    103         //evt->dumpDomains("CGenerator::configureEvent");
    104     }
    105     else
    106     {
    107          LOG(info) << "CGenerator::configureEvent"
    108                    << " no genstep (DYNAMIC RUNNING) "
    109                    ;
    110     }
    111 }


FIXED dynamic/static confusion problem, the CGenerator initializer list needs to initSource last : as that can 
change m_gensteps and m_dynamic and the rest::

     26 CGenerator::CGenerator(OpticksGen* gen, CG4* g4)
     27     :
     28     m_gen(gen), 
     29     m_ok(m_gen->getOpticks()),
     30     m_cfg(m_ok->getCfg()),
     31     m_g4(g4),
     32     m_source_code(m_gen->getSourceCode()),
     33     m_gensteps(NULL),
     34     m_dynamic(true),
     35     m_num_g4evt(1),
     36     m_photons_per_g4evt(0),
     37     m_source(initSource(m_source_code))
     38 {   
     39     init();
     40 }




OKG4Test : gensteps assert tripped at upload event
----------------------------------------------------

::


    epsilon:~ blyth$ lldb OKG4Test 

    ... after the G4 simulation ...

    2018-08-21 10:55:36.384 INFO  [501584] [OpticksViz::uploadEvent@298] OpticksViz::uploadEvent (1) DONE 
    2018-08-21 10:55:36.384 INFO  [501584] [OEvent::createBuffers@68] OEvent::createBuffers  genstep NULL nopstep 0,4,4 photon 0,4,4 source NULL record 0,10,2,4 phosel 0,1,4 recsel 0,10,1,4 sequence 0,1,2 seed 0,1,1 hit 0,4,4
    Assertion failed: (gensteps), function createBuffers, file /Users/blyth/opticks/optixrap/OEvent.cc, line 77.
    Process 99632 stopped
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
        frame #4: 0x00000001004def33 libOptiXRap.dylib`OEvent::createBuffers(this=0x000000012766b720, evt=0x000000012d5df6e0) at OEvent.cc:77
        frame #5: 0x00000001004e0721 libOptiXRap.dylib`OEvent::upload(this=0x000000012766b720, evt=0x000000012d5df6e0) at OEvent.cc:262
        frame #6: 0x00000001004e057e libOptiXRap.dylib`OEvent::upload(this=0x000000012766b720) at OEvent.cc:251
        frame #7: 0x0000000100400979 libOKOP.dylib`OpEngine::uploadEvent(this=0x000000011865d270) at OpEngine.cc:101
        frame #8: 0x0000000100104d50 libOK.dylib`OKPropagator::uploadEvent(this=0x00000001186636f0) at OKPropagator.cc:97
        frame #9: 0x00000001001049be libOK.dylib`OKPropagator::propagate(this=0x00000001186636f0) at OKPropagator.cc:74
        frame #10: 0x00000001000df411 libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe9c8) at OKG4Mgr.cc:150
        frame #11: 0x00000001000deec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe9c8) at OKG4Mgr.cc:84
        frame #12: 0x00000001000148b9 OKG4Test`main(argc=1, argv=0x00007ffeefbfeab0) at OKG4Test.cc:9
        frame #13: 0x00007fff754bb015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 9
    frame #9: 0x00000001001049be libOK.dylib`OKPropagator::propagate(this=0x00000001186636f0) at OKPropagator.cc:74
       71  	
       72  	    if(m_viz) m_hub->target();     // if not Scene targetted, point Camera at gensteps 
       73  	
    -> 74  	    uploadEvent();
       75  	
       76  	    m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries
       77  	
    (lldb) f 8
    frame #8: 0x0000000100104d50 libOK.dylib`OKPropagator::uploadEvent(this=0x00000001186636f0) at OKPropagator.cc:97
       94  	
       95  	    int npho = -1 ; 
       96  	#ifdef OPTICKS_OPTIX
    -> 97  	    npho = m_engine->uploadEvent();
       98  	#endif
       99  	    return npho ; 
       100 	}
    (lldb) 
    (lldb) f 7 
    frame #7: 0x0000000100400979 libOKOP.dylib`OpEngine::uploadEvent(this=0x000000011865d270) at OpEngine.cc:101
       98  	
       99  	unsigned OpEngine::uploadEvent()
       100 	{
    -> 101 	    return m_oevt->upload();                   // creates OptiX buffers, uploads gensteps
       102 	}
       103 	
       104 	void OpEngine::propagate()
    (lldb) 
    (lldb) f 6
    frame #6: 0x00000001004e057e libOptiXRap.dylib`OEvent::upload(this=0x000000012766b720) at OEvent.cc:251
       248 	{
       249 	    OpticksEvent* evt = m_ok->getEvent();
       250 	    assert(evt); 
    -> 251 	    return upload(evt) ;  
       252 	}
       253 	
       254 	unsigned OEvent::upload(OpticksEvent* evt)   
    (lldb) 
    (lldb) f 5
    frame #5: 0x00000001004e0721 libOptiXRap.dylib`OEvent::upload(this=0x000000012766b720, evt=0x000000012d5df6e0) at OEvent.cc:262
       259 	
       260 	    if(!m_buffers_created)
       261 	    {
    -> 262 	        createBuffers(evt);
       263 	    }
       264 	    else
       265 	    {
    (lldb) 
    (lldb) f 4
    frame #4: 0x00000001004def33 libOptiXRap.dylib`OEvent::createBuffers(this=0x000000012766b720, evt=0x000000012d5df6e0) at OEvent.cc:77
       74  	    m_buffers_created = true ; 
       75  	 
       76  	    NPY<float>* gensteps =  evt->getGenstepData() ;
    -> 77  	    assert(gensteps);
       78  	    m_genstep_buffer = m_ocontext->createBuffer<float>( gensteps, "gensteps");
       79  	    m_context["genstep_buffer"]->set( m_genstep_buffer );
       80  	    m_genstep_buf = new OBuf("genstep", m_genstep_buffer);
    (lldb) 

