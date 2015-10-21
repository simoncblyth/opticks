#pragma once
#include <cstring>

struct NSlice {
     unsigned int low ; 
     unsigned int high ; 
     unsigned int step ; 
     const char*  _description ; 

     NSlice(const char* slice, const char* delim=":");
     NSlice(unsigned int low, unsigned int high, unsigned int step=1);
     const char* description();
     unsigned int count();
};


inline NSlice::NSlice(unsigned int low, unsigned int high, unsigned int step) 
    :
    low(low),
    high(high),
    step(step),
    _description(0)
{
}


