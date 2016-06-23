
#include <plog/Log.h>

#include "OGLRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OGLRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OGLRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

