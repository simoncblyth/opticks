
#pragma once
#include "YOG_API_EXPORT.hh"

#define YOG_LOG__  {     YOG_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "YOG"), plog::get(), NULL );  } 

#define YOG_LOG_ {     YOG_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class YOG_API YOG_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

