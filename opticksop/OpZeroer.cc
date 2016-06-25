#include <cstddef>

#include "OpZeroer.hh"

// optickscore-
#include "OpticksEvent.hh"

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
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }




OpZeroer::OpZeroer(OContext* ocontext)  
   :
     m_ocontext(ocontext),
     m_evt(NULL),
     m_propagator(NULL)
{
}

void OpZeroer::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
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
    NPY<short>* record = m_evt->getRecordData(); 

    CResource r_rec( record->getBufferId(), CResource::W );

    CBufSpec s_rec = r_rec.mapGLToCUDA<short>() ;

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

    TBuf trec("trec", s_rec );

    trec.zero();

    TIMER("zeroRecordsViaOptiX"); 
}


