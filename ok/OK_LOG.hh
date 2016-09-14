
#pragma once
#include "OK_API_EXPORT.hh"

#define OK_LOG__  {     OK_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "OK"), plog::get(), NULL );  } 

#define OK_LOG_ {     OK_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class OK_API OK_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

