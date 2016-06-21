
#pragma once
#include "BRAP_API_EXPORT.hh"


#define BRAP_LOG_ {     BRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class BRAP_API BRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

