
#pragma once
#include "OKGL_API_EXPORT.hh"

#define OKGL_LOG__  {     OKGL_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "OKGL") );  } 

#define OKGL_LOG_ {     OKGL_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OKGL_API OKGL_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

