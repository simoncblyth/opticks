
#pragma once
#include "GGEO_API_EXPORT.hh"

#define GGEO_LOG__  {     GGEO_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "GGEO") );  } 

#define GGEO_LOG_ {     GGEO_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class GGEO_API GGEO_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

