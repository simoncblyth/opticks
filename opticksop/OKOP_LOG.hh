
#pragma once
#include "OKOP_API_EXPORT.hh"


#define OKOP_LOG_ {     OKOP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OKOP_API OKOP_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

