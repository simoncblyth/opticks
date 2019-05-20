#pragma once

#include <string>

class BTxt ; 

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BPropNames {
   public:
       static std::string libpath(const char* libname);
   public:
       BPropNames(const char* libname="GMaterialLib"); 
       const char* getLine(unsigned int num);
       unsigned int getNumLines();
       unsigned int getIndex(const char* line); // index of the line or UINT_MAX if not found
   private:
       void read();
   private:
       const char* m_libname ;
       BTxt*       m_txt ;  

};


#include "BRAP_TAIL.hh"

