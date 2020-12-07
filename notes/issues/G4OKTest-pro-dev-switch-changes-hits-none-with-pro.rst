G4OKTest-pro-dev-switch-changes-hits-none-with-pro
====================================================

Observe that flipping the OPTICKS_EMBEDDED_COMMANDLINE from default of "pro" 
to "dev" makes hits get reported where none are reported with "pro".

Investigate this.  Perhaps the hits are there just not reported due to no "--save"

::

    896 Comparing the commandlines, the culprit is probably : "--production":: 
    897 
    898       77 const char* G4Opticks::OPTICKS_EMBEDDED_COMMANDLINE = "OPTICKS_EMBEDDED_COMMANDLINE" ;
    899       78 const char* G4Opticks::fEmbeddedCommandLine_pro = " --compute --embedded --xanalytic --production --nosave" ;
    900       79 const char* G4Opticks::fEmbeddedCommandLine_dev = " --compute --embedded --xanalytic --save --natural --printenabled --pindex 0" ;
    901   


::

     869 int G4Opticks::propagateOpticalPhotons(G4int eventID)
     870 {
     ...
     913     if(m_gpu_propagate)
     914     {
     915         m_opmgr->setGensteps(m_gensteps);
     916 
     917         m_opmgr->propagate();     // GPU simulation is done in here 
     918 
     919         OpticksEvent* event = m_opmgr->getEvent();
     920         m_hits = event->getHitData()->clone() ;
     ^^^^^^^^^^^^^^^^^^^ needs to have been downloaded get get any hits ^^^^^^^^^^

     921         m_num_hits = m_hits->getNumItems() ;
     922 
     923         m_hits_wrapper->setPhotons( m_hits );
     924 
     925 
     926         if(!m_ok->isProduction())
     927         {
     928             // minimal g4 side instrumentation in "1st executable" 
     929             // do after propagate, so the event will have been created already
     930             m_g4hit = m_g4hit_collector->getPhoton();
     931             m_g4evt = m_opmgr->getG4Event();
     932             m_g4evt->saveHitData( m_g4hit ) ; // pass thru to the dir, owned by m_g4hit_collector ?
     933             m_g4evt->saveSourceData( m_genphotons ) ;
     934         }
     935 
     936 
     937         m_opmgr->reset();
     938         // reset : clears OpticksEvent buffers, excluding gensteps
     939         //         must clone any buffers to be retained before the reset
     940     }



okop/OpMgr.cc::

    115 void OpMgr::propagate()
    116 {
    117     LOG(LEVEL) << "\n\n[[\n\n" ;
    118 
    119     const Opticks& ok = *m_ok ;
    120    
    121     if(ok("nopropagate")) return ;
    122 
    123     assert( ok.isEmbedded() );
    124 
    125     assert( m_gensteps );
    126 
    127     bool production = m_ok->isProduction();
    128 
    129     bool compute = true ;
    130 
    131     m_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    132 
    133     m_run->createEvent(m_gensteps);
    134 
    135     m_propagator->propagate();
    136 
    137     if(ok("save"))
    138     {
    139         LOG(LEVEL) << "( save " ;
    140         m_run->saveEvent();
    141         LOG(LEVEL) << ") save " ;
    142 
    143         LOG(LEVEL) << "( ana " ;
    144         if(!production) m_hub->anaEvent();
    145         LOG(LEVEL) << ") ana " ;
    146     }
    147     else
    148     {
    149         LOG(LEVEL) << "NOT saving " ;
    150     }
    151 
    152     LOG(LEVEL) << "( postpropagate " ;
    153     m_ok->postpropagate();  // profiling 
    154     LOG(LEVEL) << ") postpropagate " ;
    155 
    156     LOG(LEVEL) << "\n\n]]\n\n" ;
    157 }


Hmm need to partially download (just hits) in production running even when not saving.::

    059 void OpPropagator::propagate()
     60 {
     61     OK_PROFILE("_OpPropagator::propagate");
     62 
     63     OpticksEvent* evt = m_hub->getEvent();
     64 
     65     assert(evt);
     66 
     67     LOG(fatal) << "evtId(" << evt->getId() << ") " << m_ok->brief()   ;
     68 
     69     uploadEvent();
     70 
     71     m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries
     72 
     73     OK_PROFILE("OpPropagator::propagate");
     74 
     75     int nhit = m_ok->isSave() ? downloadEvent() : -1 ;
     76 
     77     LOG(fatal) << "evtId(" << evt->getId() << ") DONE nhit: " << nhit    ;
     78 
     79     OK_PROFILE("OpPropagator::propagate-download");
     80 }
     81 


     91 int OpPropagator::downloadEvent()
     92 {
     93     int nhit = -1 ;
     94     nhit = m_engine->downloadEvent();
     95     return nhit ;
     96 }


    239 unsigned OpEngine::downloadEvent()
    240 {
    241     LOG(LEVEL) << "[" ;
    242     unsigned n = m_oevt->download();
    243     LOG(LEVEL) << "]" ;
    244     return n ;
    245 }


    445 /**
    446 OEvent::download
    447 -------------------
    448 
    449 In "--production" mode does not download the full event, only hits.
    450 
    451 **/
    452 
    453 unsigned OEvent::download()
    454 {
    455     if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);
    456 
    457     unsigned nhit = downloadHits();
    458     LOG(LEVEL) << " nhit " << nhit ;
    459 
    460     return nhit ;
    461 }

