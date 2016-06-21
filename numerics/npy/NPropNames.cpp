#include <cstring>
#include <cstdlib>
#include <cassert>

#include "BFile.hh"
#include "BLog.hh"

#include "NTxt.hpp"
#include "NPropNames.hpp"


NPropNames::NPropNames(const char* libname)
   :
   m_libname(strdup(libname)),
   m_txt(NULL) 
{
   read(); 
}


std::string NPropNames::libpath(const char* libname)
{
    std::string path = BFile::FormPath("$IDPATH", "GItemList", libname ); 
    return path ; 
}


void NPropNames::read()
{
    LOG(trace) << "NPropNames::read" 
               << " libname " << ( m_libname ? m_libname : "NULL" )
               ;

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
            LOG(trace) << "NPropNames::read"
                       << " path " << path ; 

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



