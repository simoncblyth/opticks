#include "NSlice.hpp"

#include <cstdlib>
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


NSlice::NSlice(const char* slice, const char* delim)
{
    // defaults
    low = 0 ;
    high = 1 ; 
    step = 1 ; 

    unsigned int i = 0 ;
    char* str = strdup(slice);   
    char* token;
    while ((token = strsep(&str, delim)))
    { 
       switch(i)
       {
          case 0:  low = atoi(token) ; break ; 
          case 1: high = atoi(token) ; break ; 
          case 2: step = atoi(token) ; break ; 
       }
       i++ ;
    }

    if(i == 1) high = low+1 ;  // when only single int provided

}
