
#pragma once
#include "OKGEO_API_EXPORT.hh"

#define OKGEO_LOG__  {     OKGEO_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKGEO"), plog::get(), NULL );  } 

#define OKGEO_LOG_ {     OKGEO_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKGEO_API OKGEO_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

