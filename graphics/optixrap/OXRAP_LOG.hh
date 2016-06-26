
#pragma once
#include "OXRAP_API_EXPORT.hh"

#define OXRAP_LOG__  {     OXRAP_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "OXRAP") );  } 

#define OXRAP_LOG_ {     OXRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OXRAP_API OXRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

