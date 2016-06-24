
#include <plog/Log.h>

#include "CUDARAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void CUDARAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void CUDARAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

