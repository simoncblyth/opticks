
#pragma once
#include "OKCORE_API_EXPORT.hh"

#define OKCORE_LOG__  {     OKCORE_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKCORE"), plog::get(), NULL );  } 

#define OKCORE_LOG_ {     OKCORE_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKCORE_API OKCORE_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

