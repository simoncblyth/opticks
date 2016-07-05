#pragma once

#include <vector>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BDir {
   public:
      static void dirlist(std::vector<std::string>& names,  const char* path);
      static void dirlist(std::vector<std::string>& basenames,  const char* path, const char* ext);

     // basenames of directories within the path directory
      static void dirdirlist(std::vector<std::string>& names,  const char* path);

};

#include "BRAP_TAIL.hh"

