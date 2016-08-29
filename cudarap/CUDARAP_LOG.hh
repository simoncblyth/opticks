
#pragma once
#include "CUDARAP_API_EXPORT.hh"

#define CUDARAP_LOG__  {     CUDARAP_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "CUDARAP"), plog::get(), NULL );  } 

#define CUDARAP_LOG_ {     CUDARAP_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class CUDARAP_API CUDARAP_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

