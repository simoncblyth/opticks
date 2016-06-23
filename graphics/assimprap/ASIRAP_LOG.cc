
#include <plog/Log.h>

#include "ASIRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void ASIRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void ASIRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

