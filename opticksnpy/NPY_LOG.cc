
#include <plog/Log.h>

#include "NPY_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void NPY_LOG::Initialize(int level, void* app1, void* app2 )
{
    PLOG_INIT(level, app1, app2);
}
void NPY_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

