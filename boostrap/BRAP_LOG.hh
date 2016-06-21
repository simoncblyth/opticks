
#pragma once
#include "BRAP_API_EXPORT.hh"

class BRAP_API BRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

