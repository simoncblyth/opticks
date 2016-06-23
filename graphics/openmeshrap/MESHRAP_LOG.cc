
#include <plog/Log.h>

#include "MESHRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void MESHRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void MESHRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

