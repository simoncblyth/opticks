
#include <plog/Log.h>

#include "OKCORE_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKCORE_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OKCORE_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

