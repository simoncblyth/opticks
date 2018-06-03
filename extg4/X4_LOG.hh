
#pragma once
#include "X4_API_EXPORT.hh"

#define X4_LOG__  {     X4_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "X4"), plog::get(), NULL );  } 

#define X4_LOG_ {     X4_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class X4_API X4_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

