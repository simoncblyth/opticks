#pragma once
#include "CSG_API_EXPORT.hh"

#define CSG_LOG__  {     CSG_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "CSG"), plog::get(), NULL );  } 

#define CSG_LOG_ {     CSG_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class CSG_API CSG_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

