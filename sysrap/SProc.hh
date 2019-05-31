#pragma once

/**
SProc
=======

macOS implementation of VirtualMemoryUsageMB of a process.

**/

#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SProc {
  public:
      static float VirtualMemoryUsageMB();
      static const char* ExecutablePath(bool basename=false); 
      static const char* ExecutableName(); 


};
