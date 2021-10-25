#pragma once
#include "CSGOPTIX_API_EXPORT.hh"

#define CSGOPTIX_LOG__  {     CSGOPTIX_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "CSGOPTIX"), plog::get(), NULL );  } 

#define CSGOPTIX_LOG_ {     CSGOPTIX_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class CSGOPTIX_API CSGOPTIX_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

