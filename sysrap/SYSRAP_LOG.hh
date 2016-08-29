
#pragma once
#include "SYSRAP_API_EXPORT.hh"

#define SYSRAP_LOG__  {     SYSRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "SYSRAP"), plog::get(), NULL );  } 

#define SYSRAP_LOG_ {     SYSRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class SYSRAP_API SYSRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

