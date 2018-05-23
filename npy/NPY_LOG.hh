
#pragma once
#include "NPY_API_EXPORT.hh"

#define NPY_LOG__  {     NPY_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "NPY"), plog::get(), NULL );  } 

#define NPY_LOG_ {     NPY_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class NPY_API NPY_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

