
#pragma once
#include "Y4CSG_API_EXPORT.hh"

#define Y4CSG_LOG__  {     Y4CSG_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "Y4CSG"), plog::get(), NULL );  } 

#define Y4CSG_LOG_ {     Y4CSG_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class Y4CSG_API Y4CSG_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

