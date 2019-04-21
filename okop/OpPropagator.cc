#include "SLog.hh"


#include "NGLM.hpp"
#include "NPY.hpp"

#include "BTimeKeeper.hh"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#include "OpPropagator.hh" // okop-
#include "OpEngine.hh"  
#include "OpTracer.hh"  


#include "PLOG.hh"
#include "OKOP_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          BTimeKeeper& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



const plog::Severity OpPropagator::LEVEL = debug ; 

OpPropagator::OpPropagator(OpticksHub* hub, OpticksIdx* idx) 
    :
    m_log(new SLog("OpPropagator::OpPropagator", "", LEVEL)),
    m_hub(hub),    
    m_idx(idx),
    m_ok(m_hub->getOpticks()),
    m_engine(new OpEngine(m_hub)),
    m_tracer(new OpTracer(m_engine,m_hub, true)),
    m_placeholder(0)
{
    (*m_log)("DONE");
}



void OpPropagator::propagate()
{
    OK_PROFILE("OpPropagator::propagate.BEG");


    OpticksEvent* evt = m_hub->getEvent();

    assert(evt);

    LOG(fatal) << "evtId(" << evt->getId() << ") " << m_ok->brief()   ;

    uploadEvent();

    m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries

    OK_PROFILE("OpPropagator::propagate.MID");

    int nhit = m_ok->isSave() ? downloadEvent() : -1 ; 

    LOG(fatal) << "evtId(" << evt->getId() << ") DONE nhit: " << nhit    ;

    OK_PROFILE("OpPropagator::propagate.END");
}



int OpPropagator::uploadEvent()
{
    int npho = -1 ; 
    npho = m_engine->uploadEvent();
    return npho ; 
}

int OpPropagator::downloadEvent()
{
    int nhit = -1 ; 
    nhit = m_engine->downloadEvent();
    return nhit ; 
}


void OpPropagator::indexEvent()
{
    m_idx->indexBoundariesHost();

    m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

    m_idx->indexSeqHost();
}


void OpPropagator::cleanup()
{
    m_engine->cleanup();
}

void OpPropagator::snap()
{
    LOG(info) << "OpPropagator::snap" ; 
    m_tracer->snap();
}



