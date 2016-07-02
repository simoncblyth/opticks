
#include <plog/Log.h>

#include "GGV_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void GGV_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void GGV_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

