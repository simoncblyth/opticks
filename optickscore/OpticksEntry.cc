#include <sstream>

#include "OpticksEntry.hh"


const char* OpticksEntry::GENERATE_ = "GENERATE" ; 
const char* OpticksEntry::TRIVIAL_  = "TRIVIAL" ; 
const char* OpticksEntry::NOTHING_ = "NOTHING" ; 
const char* OpticksEntry::SEEDTEST_ = "SEEDTEST" ; 
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
       default: name = UNKNOWN_  ; break ; 
    }
    return name ; 
}

bool OpticksEntry::isTrivial() { return m_code == 'T' ; }
bool OpticksEntry::isNothing() { return m_code == 'N' ; }

std::string OpticksEntry::description()
{
    std::stringstream ss ; 
    ss << "OpticksEntry (" << m_index << ") " << m_code ;
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



