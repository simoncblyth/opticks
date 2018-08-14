OKOP : Opticks Operations (pure compute, no viz)
===================================================

One line descriptions
-----------------------

OpMgr
   steering : contains Op[Propagator,Evt], Opticks[Hub,Idx,Gen,Run]

OpPropagator
   middle management : does little, runs engine : contains Op[Engine,Tracer], Opticks[Hub,Idx]

OpEngine
   worker : contains O[Scene,Context,Event] Op[Seeder,Zeroer,Indexer] Opticks[Hub,Entry] 

OpEvt
   minimal embedded genstep control, CFG4.CCollector does all that this does and more :
   perhaps can avoid genstep intermediary and just operate from the G4StepNPY ?
   
    
OpSeeder

OpTracer

OpZeroer

OpIndexer

OpIndexerApp


oxrap- basis classes
-------------------------

* see :doc:`../optixrap/OXRAP` 


Thoughts
----------

To some extent the OpticksHub is getting everywhere just as 
having a shared context is convenient : rather than for its functionality.  
Where that is the case probably better to replace with an OpticksCtx 
that just acts to hold on to things.


OpKernel in planning 
----------------------

Aiming for the simplest possible way to take a 
set of gensteps, pass to GPU, generate and propagate 
photons and copy back hits.   
No frills, except perhaps indexing 


::

    OpticksEvent* m_event ;   
    OpPropagator* m_propagator ;
 

Review these : see if they are doing anything not needed.

okc.OpticksEvent
    


OpMgr(Opticks* ok )
--------------------

High level steering for compute only Opticks, **only** used from::

    okop/tests/OpSnapTest
    g4ok/G4Opticks

The **only** means have free reign to change this.

::

     51    private: 
     52        SLog*          m_log ;
     53        Opticks*       m_ok ; 
     54        OpticksHub*    m_hub ;
     55        OpticksIdx*    m_idx ; 
     56        int            m_num_event ;
     57        OpticksGen*    m_gen ; 
     58        OpticksRun*    m_run ; 
     59        OpPropagator*  m_propagator ;
     60        int            m_count ;  
     61        OpEvt*         m_opevt ;  


OpticksRun

     * dual g4/ok event handling (kernel not to do this, do at higher level)
     * genstep translation using G4StepNPY m_g4step 

OpticksGen 
      



OpMgr::Propagate
~~~~~~~~~~~~~~~~~~~~~~

Notice in propagate() repetition of the interplay between 
OpPropagator.m_propagator and OpticksRun.m_run ... 
perhaps factor out into OpKernel ?  
 


OpPropagator(OpticksHub* hub, OpticksIdx* idx )
-------------------------------------------------


OpEngine(OpticksHub* hub)
---------------------------

::

     66     private:
     67        // ctor instanciated always
     68        SLog*                m_log ;
     69        OpticksHub*          m_hub ; 
     70        Opticks*             m_ok ;
     71        OScene*              m_scene ; 
     72        OContext*            m_ocontext ;
     73     private:
     74        // conditionally instanciated in init, not for isLoad isTracer 
     75        OpticksEntry*        m_entry ;
     76        OEvent*              m_oevt ;
     77        OPropagator*         m_propagator ;
     78        OpSeeder*            m_seeder ;
     79        OpZeroer*            m_zeroer ;  
     80        OpIndexer*           m_indexer ;








See Also
----------

* :doc:`../optickscore/OKCORE`
* :doc:`../opticksgeo/OKGEO`
* :doc:`../okop/OKOP`
* :doc:`../optixrap/OXRAP`
* :doc:`../thrustrap/THRAP`




