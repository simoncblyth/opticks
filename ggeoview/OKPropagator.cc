#include "OKPropagator.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#include "OKGLTracer.hh"     // optixgl-
#include "OpEngine.hh"  // opticksop-

#define GUI_ 1
#include "OpticksViz.hh"

#include "PLOG.hh"
#include "GGV_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



OKPropagator::OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz) 
    :
    m_hub(hub),    
    m_idx(idx),
    m_viz(viz),    
    m_ok(m_hub->getOpticks()),
    m_engine(new OpEngine(m_hub)),
    m_tracer(m_viz ? new OKGLTracer(m_engine,m_viz, true) : NULL )
{
    init();
    LOG(fatal) << "OKPropagator::OKPropagator DONE"
               << " viz " << viz 
                 ;
}

void OKPropagator::init()
{
}


void OKPropagator::propagate(NPY<float>* genstep)
{
    int n_gs  = genstep ? genstep->getNumItems() : -1 ;   // hmm could count photons instead of steps ?

    if( n_gs <= 0 )
    {  
         LOG(fatal) << "OKPropagator::propagate"
                     << " SKIPPING as no collected optical gensteps (ie Cerenkov or scintillation gensteps) "
                     << " or fabricated torch gensteps  "
                     ;   
         return ;   
    }


    m_hub->initOKEvent(genstep);

    if(m_viz)
    { 
        m_hub->target();             // if not Scene targetted, point Camera at gensteps 

        m_viz->uploadEvent();        // allocates GPU buffers with OpenGL glBufferData
    }


    m_engine->propagate();           // perform OptiX GPU propagation 



    m_idx->indexBoundariesHost();

    if(m_viz) m_viz->indexPresentationPrep();
   

    if(m_ok->hasOpt("save"))
    {
        if(m_viz) m_viz->downloadEvent();

        m_engine->downloadEvt();

        m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

        m_idx->indexSeqHost();

        m_hub->save();
    }
}






void OKPropagator::cleanup()
{
    m_engine->cleanup();
}





