
#pragma once
#include "ASIRAP_API_EXPORT.hh"


#define ASIRAP_LOG_ {     ASIRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class ASIRAP_API ASIRAP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

