
#include <plog/Log.h>

#include "OKCORELog.hpp"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKCORELog::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OKCORELog::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

