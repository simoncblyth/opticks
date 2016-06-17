#pragma once
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

class BRAP_API BTime {
   public:
       BTime();  

       int check();
       static void current_time(std::string& ct,  const char* tfmt, int utc);
       static std::string now(const char* tfmt,  int utc );
};





