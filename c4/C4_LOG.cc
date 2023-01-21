
#include <plog/Log.h>

#include "C4_LOG.hh"
#include "SLOG_INIT.hh"
#include "SLOG.hh"
       
void C4_LOG::Initialize(int level, void* app1, void* app2 )
{
    SLOG_INIT(level, app1, app2);
}
void C4_LOG::Check(const char* msg)
{
    SLOG_CHECK(msg);
}

