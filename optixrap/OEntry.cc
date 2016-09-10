#include <sstream>

#include "OEntry.hh"

bool OEntry::isTrivial()
{
    return m_code == 'T' ; 
}

std::string OEntry::description()
{
    std::stringstream ss ; 
    ss << "OEntry (" << m_index << ") " << m_code ;
    return ss.str();
}

unsigned OEntry::getIndex()
{
   return m_index ; 
}

OEntry::OEntry( unsigned index, char code) 
   :
   m_index(index),
   m_code(code)
{
}



