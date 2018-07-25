#include <cstring>
#include <cassert>

#include "SId.hh"

SId::SId(const char* identifiers_ )
   :
   identifiers(strdup(identifiers_)),
   idx(-1)
{
}
const char* SId::get(bool reset)
{
    if(reset) idx = -1 ; 
    idx += 1 ; 
    assert( idx < int(strlen(identifiers))) ; 

    char id[2] ; 
    id[0] = identifiers[idx] ; 
    id[1] = '\0' ; 
    return strdup(id) ; 
}


