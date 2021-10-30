#pragma once

#include <string> 
#include <map> 

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SCount 
{
    std::map<int, int>  mii ;  

    void add(int idx);  
    std::string desc() const ;     
    bool is_all(int count) const; 

}; 
