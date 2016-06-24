
#pragma once
#include "SYSRAP_API_EXPORT.hh"


#define SYSRAP_LOG_ {     SYSRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class SYSRAP_API SYSRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

