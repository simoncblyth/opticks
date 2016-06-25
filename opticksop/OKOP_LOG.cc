
#include <plog/Log.h>

#include "OKOP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKOP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OKOP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

