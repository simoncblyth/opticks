
#include <plog/Log.h>

#include "NPY_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void NPY_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void NPY_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

