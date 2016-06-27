
#pragma once
#include "CFG4_API_EXPORT.hh"

#define CFG4_LOG__  {     CFG4_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "CFG4") );  } 

#define CFG4_LOG_ {     CFG4_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class CFG4_API CFG4_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

