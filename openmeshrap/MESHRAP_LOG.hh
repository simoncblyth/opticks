
#pragma once
#include "MESHRAP_API_EXPORT.hh"

#define MESHRAP_LOG__  {     MESHRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "MESHRAP"), plog::get(), NULL );  } 

#define MESHRAP_LOG_ {     MESHRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class MESHRAP_API MESHRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

