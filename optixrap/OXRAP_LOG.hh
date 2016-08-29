
#pragma once
#include "OXRAP_API_EXPORT.hh"

#define OXRAP_LOG__  {     OXRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OXRAP"), plog::get(), NULL );  } 

#define OXRAP_LOG_ {     OXRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OXRAP_API OXRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

