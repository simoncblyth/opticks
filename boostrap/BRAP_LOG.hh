
#pragma once
#include "BRAP_API_EXPORT.hh"

#define BRAP_LOG__  {     BRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "BRAP"), plog::get(), NULL );  } 

#define BRAP_LOG_ {     BRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class BRAP_API BRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

