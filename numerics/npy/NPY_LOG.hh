
#pragma once
#include "NPY_API_EXPORT.hh"


#define NPY_LOG_ {     NPY_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class NPY_API NPY_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

