
#include <plog/Log.h>

#include "GGEO_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void GGEO_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void GGEO_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

