#pragma once

/**
SGDML
=======

GDML style generation of a unique name for an object using the
memory location pointer.

**/


#include <string>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SGDML {
  public:
      // based on G4GDMLWrite::GenerateName 
      static std::string GenerateName(const char* name, const void* const ptr, bool addPointerToName=true );

};


