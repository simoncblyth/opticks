
#pragma once
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OKCORE_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

