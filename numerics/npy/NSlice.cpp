#include "NSlice.hpp"
#include <cstdio>
#include <cstring>

const char* NSlice::description()
{
    if(!_description)
    { 
        char desc[128];
        snprintf(desc, 128, "NSlice  %5d : %5d : %5d ", low, high, step );
        _description = strdup(desc) ;
    }
    return _description ; 
}
