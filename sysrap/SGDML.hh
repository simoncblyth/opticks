#pragma once
#include <string>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SGDML {
  public:
      // based on G4GDMLWrite::GenerateName 
      static std::string GenerateName(const char* name, const void* const ptr, bool addPointerToName=true );

};


