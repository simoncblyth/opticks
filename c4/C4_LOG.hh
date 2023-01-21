#pragma once
#include "C4_API_EXPORT.hh"

#define C4_LOG__  {     C4_LOG::Initialize(SLOG::instance->prefixlevel_parse( info, "C4"), plog::get(), NULL );  } 

#define C4_LOG_ {     C4_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 

class C4_API C4_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

