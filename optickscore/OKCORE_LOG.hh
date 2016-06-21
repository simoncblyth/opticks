
#pragma once
#include "OKCORE_API_EXPORT.hh"


#define OKCORE_LOG_ {     OKCORE_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); } 

class OKCORE_API OKCORE_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

