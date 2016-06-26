
#pragma once
#include "CUDARAP_API_EXPORT.hh"

#define CUDARAP_LOG__  {     CUDARAP_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "CUDARAP") );  } 

#define CUDARAP_LOG_ {     CUDARAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class CUDARAP_API CUDARAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

