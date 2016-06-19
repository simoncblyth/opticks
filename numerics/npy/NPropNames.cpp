#include "NPropNames.hpp"
#include "NTxt.hpp"
#include "BLog.hh"

#include <cstring>
#include <cstdlib>
#include <cassert>


NPropNames::NPropNames(const char* libname)
   :
   m_libname(strdup(libname)),
   m_txt(NULL) 
{
   read(); 
}


std::string NPropNames::libpath(const char* libname)
{
    char* idp = getenv("IDPATH") ;
    char path[256];
    snprintf(path, 256, "%s/GItemList/%s.txt", idp, libname );
    return path ; 
}


void NPropNames::read()
{
    if(!m_txt)
    {
        if(strlen(m_libname) > 0 && m_libname[0] == '/' )
        {
            // absolute path for testing  
            m_txt = new NTxt(m_libname);
        } 
        else
        {
            // GItemList name like GMaterialLib 
            std::string path = libpath(m_libname) ; 
            m_txt = new NTxt(path.c_str());
        }
    }
    m_txt->read();
}


const char* NPropNames::getLine(unsigned int num)
{
   return m_txt ? m_txt->getLine(num) : NULL ; 
}
unsigned int  NPropNames::getNumLines()
{
   return m_txt ? m_txt->getNumLines() : 0 ;
}

unsigned int  NPropNames::getIndex(const char* line)
{
   assert(m_txt); 
   return m_txt->getIndex(line) ;
}



