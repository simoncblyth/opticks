#include <cstddef>

#include "OpZeroer.hh"

// optickscore-
#include "OpticksEvent.hh"

// opticksgeo-
#include "OpticksHub.hh"

// npy-
#include "Timer.hpp"
#include "PLOG.hh"
#include "NPY.hpp"

// cudawrap-
#include "CResource.hh"
#include "CBufSpec.hh"

// thrustrap-
#include "TBuf.hh"

// optixrap-
#include "OContext.hh"
#include "OPropagator.hh"
#include "OBuf.hh"


#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }




OpZeroer::OpZeroer(OpticksHub* hub, OContext* ocontext)  
   :
     m_hub(hub),
     m_ocontext(ocontext),
     m_propagator(NULL)
{
}

void OpZeroer::setPropagator(OPropagator* propagator)
{
    m_propagator = propagator ; 
}  


void OpZeroer::zeroRecords()
{
    LOG(info)<<"OpZeroer::zeroRecords" ;

    if( m_ocontext->isInterop() )
    {    
        zeroRecordsViaOpenGL();
    }    
    else if ( m_ocontext->isCompute() )
    {    
        zeroRecordsViaOptiX();
    }    
}


void OpZeroer::zeroRecordsViaOpenGL()
{
    OpticksEvent* evt = m_hub->getEvent();

    NPY<short>* record = evt->getRecordData(); 

    CResource r_rec( record->getBufferId(), CResource::W );

    CBufSpec s_rec = r_rec.mapGLToCUDA<short>() ;

    s_rec.Summary("OpZeroer::zeroRecordsViaOpenGL(CBufSpec)s_rec");

    TBuf trec("trec", s_rec );

    trec.zero();

    r_rec.unmapGLToCUDA();

    TIMER("zeroRecordsViaOpenGL"); 
}


void OpZeroer::zeroRecordsViaOptiX()
{
    assert(m_propagator);
 
    OBuf* record = m_propagator->getRecordBuf() ;

    CBufSpec s_rec = record->bufspec();

    s_rec.Summary("OpZeroer::zeroRecordsViaOptiX(CBufSpec)s_rec");

    TBuf trec("trec", s_rec );

    trec.zero();

    TIMER("zeroRecordsViaOptiX"); 
}


