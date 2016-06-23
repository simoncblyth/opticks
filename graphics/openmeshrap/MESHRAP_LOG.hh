
#pragma once
#include "MESHRAP_API_EXPORT.hh"


#define MESHRAP_LOG_ {     MESHRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class MESHRAP_API MESHRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

