#include <sstream>
#include <cstring>
#include <cassert>

#include "SId.hh"

SId::SId(const char* identifiers_ )
   :
   identifiers(strdup(identifiers_)),
   idx(-1),
   cycle(0)
{
}

void SId::reset()
{
    idx = -1 ; 
    cycle = 0 ; 
}

const char* SId::get(bool reset_)
{
    if(reset_) reset();  

    if( idx + 1 == int(strlen(identifiers)) ) 
    { 
        cycle += 1 ; 
        idx = -1 ; 
    }

    idx += 1 ; 
    assert( idx < int(strlen(identifiers))) ; 

    std::stringstream ss ; 
    ss << identifiers[idx] ;
    if( cycle > 0 ) ss << cycle ; 
         
    std::string s = ss.str() ; 

    return strdup(s.c_str()) ; 
}


