
#pragma once
#include "OKGL_API_EXPORT.hh"

#define OKGL_LOG__  {     OKGL_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OKGL"), plog::get(), NULL );  } 

#define OKGL_LOG_ {     OKGL_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OKGL_API OKGL_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

