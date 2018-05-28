
#pragma once
#include "G4OK_API_EXPORT.hh"

#define G4OK_LOG__  {     G4OK_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "G4OK"), plog::get(), NULL );  } 

#define G4OK_LOG_ {     G4OK_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); } 
class G4OK_API G4OK_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

