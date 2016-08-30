
#include <plog/Log.h>

#include "OKG4_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void OKG4_LOG::Initialize(int level, void* app1, void* app2 )
{
    PLOG_INIT(level, app1, app2);
}
void OKG4_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

