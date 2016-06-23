
#include <plog/Log.h>

#include "OKGEO_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKGEO_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OKGEO_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

