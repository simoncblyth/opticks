
#pragma once
#include "THRAP_API_EXPORT.hh"

#define THRAP_LOG__  {     THRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "THRAP"), plog::get(), NULL );  } 

#define THRAP_LOG_ {     THRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class THRAP_API THRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

