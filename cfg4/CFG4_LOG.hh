
#pragma once
#include "CFG4_API_EXPORT.hh"

#define CFG4_LOG__  {     CFG4_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "CFG4"), plog::get(), NULL );  } 

#define CFG4_LOG_ {     CFG4_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class CFG4_API CFG4_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

