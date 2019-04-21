#include "OKG4Mgr.hh"

class NConfigurable ; 

#include "SLog.hh"
#include "BTimeKeeper.hh"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksRun.hh"    

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    

#include "OKPropagator.hh"  // ok-

#define GUI_ 1
#include "OpticksViz.hh"

#include "CG4.hh"
#include "CGenerator.hh"

#include "PLOG.hh"
#include "OKG4_BODY.hh"

int OKG4Mgr::rc() const 
{
    return m_ok->rc();
}


OKG4Mgr::OKG4Mgr(int argc, char** argv) 
    :
    m_log(new SLog("OKG4Mgr::OKG4Mgr")),
    m_ok(new Opticks(argc, argv)),  
    m_run(m_ok->getRun()),
    m_hub(new OpticksHub(m_ok)),            // configure, loadGeometry and setupInputGensteps immediately
    m_load(m_ok->isLoad()),
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // huh : m_gen should be in change of the number of events ? 
    m_gen(m_hub->getGen()),
    m_g4(m_load ? NULL : new CG4(m_hub)),   // configure and initialize immediately 
    m_generator( m_load ? NULL : m_g4->getGenerator()),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz))
{
    (*m_log)("DONE");
}

OKG4Mgr::~OKG4Mgr()
{
    cleanup();
}


void OKG4Mgr::propagate()
{
    const Opticks& ok = *m_ok ;

    if(m_load)
    {   
         m_run->loadEvent(); 

         if(m_viz) 
         {   
             m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

             m_viz->uploadEvent();      // not needed when propagating as event is created directly on GPU

             m_viz->indexPresentationPrep();
         }   
    }   
    else if(ok("nopropagate"))
    {   
        LOG(info) << "--nopropagate/-P" ;
    }   
    else if(m_num_event > 0)
    {
        for(int i=0 ; i < m_num_event ; i++) 
        {   
            m_run->createEvent(i);

            propagate_();

            if(ok("save"))
            {
                m_run->saveEvent();
                m_hub->anaEvent();
            }

            m_run->resetEvent();

        }
        m_ok->postpropagate();
    }   
}




/**
OKG4Mgr::propagate_
---------------------

Hmm propagate implies just photons to me, so this name 
is misleading as it does a G4 beamOn with the hooked up 
CSource subclass providing the primaries, which can be 
photons but not necessarily. 

Normally the G4 propagation is done first, because 
gensteps eg from G4Gun can then be passed to Opticks.
However with RNG-aligned testing using "--align" option
which uses emitconfig CPU generated photons there is 
no need to do G4 first. Actually it is more convenient
for Opticks to go first in order to allow access to the ucf.py 
parsed  kernel pindex log during lldb python scripted G4 debugging.
 
Hmm it would be cleaner if m_gen was in charge if the branching 
here as its kinda similar to initSourceCode.


Notice the different genstep handling between this and OKMgr 
because this has G4 available, so gensteps can come from the
horses mouth.



**/

void OKG4Mgr::propagate_()
{ 
    bool align = m_ok->isAlign();


    if(m_generator->hasGensteps())   // TORCH
    {
         NPY<float>* gs = m_generator->getGensteps() ;
         m_run->setGensteps(gs); 

         if(align)
             m_propagator->propagate();
          
         m_g4->propagate();
    }
    else   // no-gensteps : G4GUN or PRIMARYSOURCE
    {
         NPY<float>* gs = m_g4->propagate() ;

         if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ; 
         assert(gs);

         m_run->setGensteps(gs); 
    }
            
    if(!align)
        m_propagator->propagate();
}




void OKG4Mgr::visualize()
{
    if(m_viz) m_viz->visualize();
}

void OKG4Mgr::cleanup()
{
#ifdef OPTICKS_OPTIX
    m_propagator->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
    m_g4->cleanup(); 
}


/**
   
    tpmt-- --okg4 --live --compute --debugger

**/

