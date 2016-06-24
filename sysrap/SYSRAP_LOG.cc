
#include <plog/Log.h>

#include "SYSRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void SYSRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void SYSRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

