
#include <plog/Log.h>

#include "CFG4_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void CFG4_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void CFG4_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

