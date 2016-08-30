
#pragma once
#include "OKG4_API_EXPORT.hh"

#define OKG4_LOG__  {     OKG4_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKG4"), plog::get(), NULL );  } 

#define OKG4_LOG_ {     OKG4_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKG4_API OKG4_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

