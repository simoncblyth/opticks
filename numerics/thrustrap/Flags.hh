#pragma once

#include <map>
#include <string>

class Flags {
   public:
       Flags();
       void read(const char* path);
       void dump(const char* msg="Flags::dump");
       std::string getSequenceString(unsigned long long seq);

   private:
        std::map<std::string, unsigned int>  m_name2code ; 
        std::map<unsigned int, std::string>  m_code2name ; 

};


inline Flags::Flags()
{
}

