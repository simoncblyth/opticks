#include <cstring>
#include <cstdlib>
#include <cassert>

#include "BFile.hh"

#include "BTxt.hh"
#include "BPropNames.hh"

#include "PLOG.hh"

BPropNames::BPropNames(const char* libname)
   :
   m_libname(strdup(libname)),
   m_txt(NULL) 
{
   read(); 
}


std::string BPropNames::libpath(const char* libname)
{
    std::string path = BFile::FormPath("$IDPATH", "GItemList", libname ); 
    return path ; 
}


void BPropNames::read()
{
    LOG(verbose) << "BPropNames::read" 
               << " libname " << ( m_libname ? m_libname : "NULL" )
               ;

    if(!m_txt)
    {

        if(strlen(m_libname) > 0 && m_libname[0] == '/' )
        {
            // absolute path for testing  
            m_txt = new BTxt(m_libname);
        } 
        else
        {
            // GItemList name like GMaterialLib 
            std::string path = libpath(m_libname) ; 
            LOG(verbose) << "BPropNames::read"
                       << " path " << path ; 

            m_txt = new BTxt(path.c_str());
        }
    }
    m_txt->read();
}


const char* BPropNames::getLine(unsigned int num)
{
   return m_txt ? m_txt->getLine(num) : NULL ; 
}
unsigned int  BPropNames::getNumLines()
{
   return m_txt ? m_txt->getNumLines() : 0 ;
}

unsigned int  BPropNames::getIndex(const char* line)
{
   assert(m_txt); 
   return m_txt->getIndex(line) ;
}



