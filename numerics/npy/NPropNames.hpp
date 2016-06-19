#pragma once

#include <string>

class NTxt ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NPropNames {
   public:
       static std::string libpath(const char* libname);
   public:
       NPropNames(const char* libname="GMaterialLib"); 
       const char* getLine(unsigned int num);
       unsigned int getNumLines();
       unsigned int getIndex(const char* line); // index of the line or UINT_MAX if not found
   private:
       void read();
   private:
       const char* m_libname ;
       NTxt*       m_txt ;  

};


#include "NPY_TAIL.hh"

