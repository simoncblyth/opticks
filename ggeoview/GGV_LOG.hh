
#pragma once
#include "GGV_API_EXPORT.hh"

#define GGV_LOG__  {     GGV_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "GGV") );  } 

#define GGV_LOG_ {     GGV_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class GGV_API GGV_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

