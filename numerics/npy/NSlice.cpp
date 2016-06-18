
#include "NPY_FLAGS.hh"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include "NSlice.hpp"


NSlice::NSlice(unsigned int low, unsigned int high, unsigned int step) 
    :
    low(low),
    high(high),
    step(step),
    _description(0)
{
}



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


unsigned int NSlice::count()
{
    unsigned int n = 0 ;
    for(unsigned int i=low ; i < high ; i+=step ) n+=1 ; 
    return n ;  
}


NSlice::NSlice(const char* slice, const char* delim)
{
    // defaults
    low = 0 ;
    high = 1 ; 
    step = 1 ; 
    _description = 0 ; 


    std::vector<std::string> elem ;
    boost::split(elem, slice, boost::is_any_of(delim));
    unsigned int size = elem.size();

    if(size > 0) low = boost::lexical_cast<unsigned int>(elem[0]);
    if(size > 1) high = boost::lexical_cast<unsigned int>(elem[1]);
    if(size > 2) step = boost::lexical_cast<unsigned int>(elem[2]);

    if(size == 1) high = low + 1 ;  // only provided low


/*
   // strsep has portability issues
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
*/




}
