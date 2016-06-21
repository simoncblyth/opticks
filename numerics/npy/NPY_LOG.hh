
#pragma once
#include "NPY_API_EXPORT.hh"

class NPY_API NPY_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

