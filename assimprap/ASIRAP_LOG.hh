
#pragma once
#include "ASIRAP_API_EXPORT.hh"

#define ASIRAP_LOG__  {     ASIRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "ASIRAP"), plog::get(), NULL );  } 

#define ASIRAP_LOG_ {     ASIRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class ASIRAP_API ASIRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

