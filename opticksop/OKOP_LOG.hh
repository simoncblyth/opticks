
#pragma once
#include "OKOP_API_EXPORT.hh"

#define OKOP_LOG__  {     OKOP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKOP"), plog::get(), NULL );  } 

#define OKOP_LOG_ {     OKOP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKOP_API OKOP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

