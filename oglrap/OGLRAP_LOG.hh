
#pragma once
#include "OGLRAP_API_EXPORT.hh"

#define OGLRAP_LOG__  {     OGLRAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OGLRAP"), plog::get(), NULL );  } 

#define OGLRAP_LOG_ {     OGLRAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OGLRAP_API OGLRAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

