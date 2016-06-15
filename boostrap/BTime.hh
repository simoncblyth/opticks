#pragma once
#include <string>

class BTime {
   public:
       static void current_time(std::string& ct,  const char* tfmt, int utc);
       static std::string now(const char* tfmt,  int utc );
};





