#pragma once

#include <bitset>
#include "SYSRAP_API_EXPORT.hh"

template<unsigned N>
struct SYSRAP_API SEnabled
{
    std::bitset<N>* enabled ; 
    SEnabled(const char* spec); 
    bool isEnabled(unsigned idx) const ; 
}; 


