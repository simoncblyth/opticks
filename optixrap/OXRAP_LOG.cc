
#include <plog/Log.h>

#include "OXRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OXRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OXRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

