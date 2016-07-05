#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template <typename T>
class BRAP_API BHex {
   public:
       static T hex_lexical_cast(const char* in);
       static std::string as_hex(T in);
       static std::string as_dec(T in);
   public:
       BHex(T in);    
       std::string as_hex(); 
       std::string as_dec(); 
   private:
        T m_in ; 
};


#include "BRAP_TAIL.hh"

