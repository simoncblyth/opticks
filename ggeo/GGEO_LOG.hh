
#pragma once
#include "GGEO_API_EXPORT.hh"

#define GGEO_LOG__  {     GGEO_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "GGEO"), plog::get(), NULL );  } 

#define GGEO_LOG_ {     GGEO_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class GGEO_API GGEO_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

