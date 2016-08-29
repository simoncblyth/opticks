
#pragma once
#include "GGV_API_EXPORT.hh"

#define GGV_LOG__  {     GGV_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "GGV"), plog::get(), NULL );  } 

#define GGV_LOG_ {     GGV_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class GGV_API GGV_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

