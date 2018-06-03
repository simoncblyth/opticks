
#include <plog/Log.h>

#include "X4_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void X4_LOG::Initialize(int level, void* app1, void* app2 )
{
    PLOG_INIT(level, app1, app2);
}
void X4_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

