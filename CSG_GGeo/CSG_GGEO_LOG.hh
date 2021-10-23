#pragma once
#include "CSG_GGEO_API_EXPORT.hh"

#define CSG_GGEO_LOG__  {     CSG_GGEO_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "CSG_GGEO"), plog::get(), NULL );  } 

#define CSG_GGEO_LOG_ {     CSG_GGEO_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 

class CSG_GGEO_API CSG_GGEO_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

