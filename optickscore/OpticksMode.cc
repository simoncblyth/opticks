#include <cassert>
#include <cstring>
#include <sstream>

#include "SSys.hh"
#include "OpticksMode.hh"

const char* OpticksMode::UNSET_MODE_  = "UNSET_MODE" ;
const char* OpticksMode::COMPUTE_MODE_  = "COMPUTE_MODE" ;
const char* OpticksMode::INTEROP_MODE_  = "INTEROP_MODE" ;
const char* OpticksMode::CFG4_MODE_     = "CFG4_MODE" ;

bool OpticksMode::isCompute()
{
    return (m_mode & COMPUTE_MODE) != 0 ;  
}
bool OpticksMode::isInterop()
{
    return (m_mode & INTEROP_MODE) != 0 ;  
}
bool OpticksMode::isCfG4()
{
    return (m_mode & CFG4_MODE) != 0  ; 
}

std::string OpticksMode::description()
{
    std::stringstream ss ; 

    if(isCompute()) ss << COMPUTE_MODE_ ; 
    if(isInterop()) ss << INTEROP_MODE_ ; 
    if(isCfG4())    ss << CFG4_MODE_ ; 

    return ss.str();
}

unsigned int OpticksMode::Parse(const char* tag)
{
    unsigned int mode = UNSET_MODE  ; 
    if(     strcmp(tag, INTEROP_MODE_)==0)  mode = INTEROP_MODE ; 
    else if(strcmp(tag, COMPUTE_MODE_)==0)  mode = COMPUTE_MODE ; 
    else if(strcmp(tag, CFG4_MODE_)==0)     mode = CFG4_MODE ; 
    return mode ; 
}

OpticksMode::OpticksMode(const char* tag)
  :
    m_mode(Parse(tag))
{
}

OpticksMode::OpticksMode(bool compute_requested) 
   : 
   m_mode(UNSET_MODE)
{
    if(SSys::IsRemoteSession())
    {
        m_mode = COMPUTE_MODE ; 
    }
    else
    {
        m_mode = compute_requested ? COMPUTE_MODE : INTEROP_MODE ;
    }
}

void OpticksMode::setOverride(unsigned int mode)
{
   assert( mode == CFG4_MODE );
   m_mode = mode ;
}





