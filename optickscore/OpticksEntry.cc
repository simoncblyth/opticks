#include <sstream>

#include "OpticksEntry.hh"
#include "OpticksCfg.hh"

const char* OpticksEntry::GENERATE_ = "GENERATE" ; 
const char* OpticksEntry::TRIVIAL_  = "TRIVIAL" ; 
const char* OpticksEntry::NOTHING_ = "NOTHING" ; 
const char* OpticksEntry::SEEDTEST_ = "SEEDTEST" ; 
const char* OpticksEntry::TRACETEST_ = "TRACETEST" ; 
const char* OpticksEntry::ZRNGTEST_ = "ZRNGTEST" ; 
const char* OpticksEntry::UNKNOWN_  = "UNKNOWN?" ; 

const char*  OpticksEntry::Name(char code)
{
    const char* name = NULL ; 
    switch(code)
    {
       case 'G':name = GENERATE_ ; break ; 
       case 'T':name = TRIVIAL_  ; break ; 
       case 'N':name = NOTHING_  ; break ; 
       case 'S':name = SEEDTEST_ ; break ; 
       case 'R':name = TRACETEST_ ; break ; 
       case 'Z':name = ZRNGTEST_ ; break ; 
       default: name = UNKNOWN_  ; break ; 
    }
    return name ; 
}


char OpticksEntry::CodeFromConfig(OpticksCfg<Opticks>* cfg)  
{
    char code ;
    if(     cfg->hasOpt("trivial"))   code = 'T' ; 
    else if(cfg->hasOpt("nothing"))   code = 'N' ; 
    else if(cfg->hasOpt("dumpseed"))  code = 'D' ; 
    else if(cfg->hasOpt("seedtest"))  code = 'S' ; 
    else if(cfg->hasOpt("tracetest")) code = 'R' ; 
    else if(cfg->hasOpt("zrngtest"))  code = 'Z' ; 
    else                              code = 'G' ; 
    return code ;
} 


bool OpticksEntry::isTraceTest() { return m_code == 'R' ; }
bool OpticksEntry::isTrivial() { return m_code == 'T' ; }
bool OpticksEntry::isNothing() { return m_code == 'N' ; }

std::string OpticksEntry::description() const 
{
    std::stringstream ss ; 
    ss << "OpticksEntry (" << m_index << ") " << m_code ;
    return ss.str();
}


std::string OpticksEntry::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " index " << m_index 
       << " code " << m_code
       << " name " << Name(m_code)
       ;
    return ss.str();
}




unsigned OpticksEntry::getIndex()
{
   return m_index ; 
}

const char* OpticksEntry::getName()
{
    return Name(m_code) ; 
}


OpticksEntry::OpticksEntry( unsigned index, char code) 
   :
   m_index(index),
   m_code(code)
{
}



