
#include "timeutil.hpp"
#include "OContext.hh"
#include "OTimes.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



void OContext::init()
{
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    //m_context->setPrintLaunchIndex(0,0,0);
    m_context->setStackSize( 2180 ); // TODO: make externally configurable, and explore performance implications

    m_context->setEntryPointCount( getNumEntryPoint() );  
    m_context->setRayTypeCount( getNumRayType() );
    m_top = m_context->createGroup();
}


void OContext::launch(unsigned int entry, unsigned int width, unsigned int height, OTimes* times)
{
    LOG(info)<< "OContext::launch";

    double t0,t1,t2,t3,t4 ; 

    t0 = getRealTime();
    m_context->validate();
    t1 = getRealTime();
    m_context->compile();
    t2 = getRealTime();
    //m_context->launch( entry, 0); 
    m_context->launch( entry, 0, 0); 
    t3 = getRealTime();
    m_context->launch( entry, width, height ); 
    t4 = getRealTime();

    if(times)
    {
        times->count     += 1 ; 
        times->validate  += t1 - t0 ;
        times->compile   += t2 - t1 ; 
        times->prelaunch += t3 - t2 ; 
        times->launch    += t4 - t3 ; 
    }
}






