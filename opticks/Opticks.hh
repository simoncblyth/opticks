#pragma once

#include <string>
#include <vector>

class Opticks {
   public:
       Opticks();

   public:
       // methods required by Cfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);

   private:
       void init();
};

inline Opticks::Opticks()
{
    init();
}


