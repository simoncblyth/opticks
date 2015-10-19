
#include "OContext.hh"


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



