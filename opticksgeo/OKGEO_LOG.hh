
#pragma once
#include "OKGEO_API_EXPORT.hh"

#define OKGEO_LOG__  {     OKGEO_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "OKGEO") );  } 

#define OKGEO_LOG_ {     OKGEO_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OKGEO_API OKGEO_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

