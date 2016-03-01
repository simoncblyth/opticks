#pragma once

#include <string>
#include <cstring>

class NTxt ; 

class NPropNames {
   public:
       static std::string libpath(const char* libname);
   public:
       NPropNames(const char* libname="GMaterialLib"); 
       const char* getLine(unsigned int num);
       unsigned int getNumLines();
   private:
       void read();
   private:
       const char* m_libname ;
       NTxt*       m_txt ;  

};

inline NPropNames::NPropNames(const char* libname)
   :
   m_libname(strdup(libname)),
   m_txt(NULL) 
{
   read(); 
}

