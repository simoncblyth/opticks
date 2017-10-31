propagating_CPU_side_input_photons
=====================================

centralize setNumPhotonsPerG4Event ?
--------------------------------------


What does torch step need to know this ? Tis it not a CG4 detail ?

::

    simon:opticks blyth$ opticks-find setNumPhotonsPerG4Event
    ./cfg4/CGenerator.cc:void CGenerator::setNumPhotonsPerG4Event(unsigned num)
    ./cfg4/CGenerator.cc:        evt->setNumPhotonsPerG4Event(getNumPhotonsPerG4Event()) ; 
    ./cfg4/CGenerator.cc:    setNumPhotonsPerG4Event( torch->getNumPhotonsPerG4Event()); 
    ./cfg4/CGenerator.cc:    setNumPhotonsPerG4Event(0); 
    ./optickscore/Opticks.cc:    torchstep->setNumPhotonsPerG4Event(photons_per_g4event);
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setNumPhotonsPerG4Event(unsigned int n)
    ./cfg4/CGenerator.hh:       void setNumPhotonsPerG4Event(unsigned num);
    ./optickscore/OpticksEvent.hh:       void setNumPhotonsPerG4Event(unsigned int n);
    ./opticksnpy/GenstepNPY.cpp:void GenstepNPY::setNumPhotonsPerG4Event(unsigned int n)
    ./opticksnpy/TorchStepNPY.cpp:void TorchStepNPY::setNumPhotonsPerG4Event(unsigned int n)
    ./opticksnpy/GenstepNPY.hpp:       void setNumPhotonsPerG4Event(unsigned int n);
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ 




Machinery revolves around gensteps
-----------------------------------


* OpticksRun::setGensteps with gensteps from OpticksGen 
  currently done at the highest level...

::

    067 void OKMgr::propagate()
     68 {
     69     const Opticks& ok = *m_ok ;
     70 
     71     if(ok("nopropagate")) return ;
     72 
     73     bool production = m_ok->isProduction();
     74 
     75     if(ok.isLoad())
     ..
     88     else if(m_num_event > 0)
     89     {
     90         for(int i=0 ; i < m_num_event ; i++)
     91         {
     92             m_run->createEvent(i);
     93 
     94             m_run->setGensteps(m_gen->getInputGensteps());
     95 
     96             m_propagator->propagate();
     97 
     98             if(ok("save"))
     99             {
    100                 m_run->saveEvent();
    101                 if(!production) m_hub->anaEvent();
    102             }
    103 
    104             m_run->resetEvent();
    105         }
    106 
    107         m_ok->postpropagate();
    108     }
    109 }




::

     56 void OKPropagator::propagate()
     57 {
     58     OK_PROFILE("OKPropagator::propagate.BEG");
     59 
     60 
     61     OpticksEvent* evt = m_hub->getEvent();
     62 
     63     assert(evt);
     64 
     65     LOG(fatal) << "OKPropagator::propagate(" << evt->getId() << ") " << m_ok->brief()   ;
     66 
     67     if(m_viz) m_hub->target();     // if not Scene targetted, point Camera at gensteps 
     68 
     69     uploadEvent();
     70 
     71     m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries
     72 
     73     OK_PROFILE("OKPropagator::propagate.MID");
     74 
     75     if(m_viz) m_viz->indexPresentationPrep();
     76 
     77     int nhit = m_ok->isSave() ? downloadEvent() : -1 ;
     78 
     79     LOG(fatal) << "OKPropagator::propagate(" << evt->getId() << ") DONE nhit: " << nhit    ;
     80 
     81     OK_PROFILE("OKPropagator::propagate.END");
     82 }



::

     86 int OKPropagator::uploadEvent()
     87 {
     88     if(m_viz) m_viz->uploadEvent();
     //
     //    passing OpenGL buffers to the renderers
     //
     89 
     90     int npho = -1 ;
     91 #ifdef WITH_OPTIX
     92     npho = m_engine->uploadEvent();
     93 #endif
     94     return npho ;
     95 }

::

    274 void OpticksViz::uploadEvent()
    275 {
    276     if(m_hub->hasOpt("nooptix|noevent")) return ;
    277 
    278     m_composition->update();
    279 
    280     OpticksEvent* evt = m_run->getCurrentEvent() ;
    281 
    282     uploadEvent(evt);
    283 }
    284 
    285 void OpticksViz::uploadEvent(OpticksEvent* evt)
    286 {
    287     LOG(info) << "OpticksViz::uploadEvent (" << evt->getId() << ")"  ;
    288 
    289     m_scene->upload(evt);
    290 
    291     if(m_hub->hasOpt("dbguploads"))
    292         m_scene->dump_uploads_table("OpticksViz::uploadEvent(--dbguploads)");
    293 
    294     LOG(info) << "OpticksViz::uploadEvent (" << evt->getId() << ") DONE "  ;
    295 }



okop/OpEngine::

     90 unsigned OpEngine::uploadEvent()
     91 {
     92     return m_oevt->upload();                   // creates OptiX buffers, uploads gensteps
     93 }


oxrap/OEvent::

    197 unsigned OEvent::upload()
    198 {
    199     OpticksEvent* evt = m_hub->getEvent();
    200     assert(evt);
    201     return upload(evt) ;
    202 }
    203 
    204 unsigned OEvent::upload(OpticksEvent* evt)
    205 {
    206     OK_PROFILE("_OEvent::upload");
    207     LOG(debug)<<"OEvent::upload id " << evt->getId()  ;
    208     setEvent(evt);
    209 
    210     if(!m_buffers_created)
    211     {
    212         createBuffers(evt);
    213     }
    214     else
    215     {
    216         resizeBuffers(evt);
    217     }
    218     unsigned npho = uploadGensteps(evt);
    219 
    220     LOG(debug)<<"OEvent::upload id " << evt->getId() << " DONE "  ;
    221 
    222     OK_PROFILE("OEvent::upload");
    223 
    224     return npho ;
    225 }
     

::

    228 unsigned OEvent::uploadGensteps(OpticksEvent* evt)
    229 {
    230     NPY<float>* gensteps =  evt->getGenstepData() ;
    231 
    232     unsigned npho = evt->getNumPhotons();
    233 
    234     if(m_ocontext->isCompute())
    235     {
    236         LOG(info) << "OEvent::uploadGensteps (COMPUTE) id " << evt->getId() << " " << gensteps->getShapeString() << " -> " << npho  ;
    237         OContext::upload<float>(m_genstep_buffer, gensteps);
    238     }
    239     else if(m_ocontext->isInterop())
    240     {
    241         assert(gensteps->getBufferId() > 0);
    242         LOG(info) << "OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId " << gensteps->getBufferId()  ;
    243     }
    244     return npho ;
    245 }


