
#pragma once
#include "THRAP_API_EXPORT.hh"


#define THRAP_LOG_ {     THRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class THRAP_API THRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

