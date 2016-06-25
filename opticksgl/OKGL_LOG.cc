
#include <plog/Log.h>

#include "OKGL_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKGL_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void OKGL_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

