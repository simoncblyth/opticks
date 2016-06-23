
#pragma once
#include "OGLRAP_API_EXPORT.hh"


#define OGLRAP_LOG_ {     OGLRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OGLRAP_API OGLRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

