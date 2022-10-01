#include <plog/Log.h>

#include "DEMO_LOG.hh"
       
void DEMO_LOG::Initialize(plog::Severity level, plog::IAppender* app )
{
    //plog::Logger<0>& logr = plog::init(level, app ); 
    plog::init(level, app ); 
}



