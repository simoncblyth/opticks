#include "BMap.hh"

#include <iostream>

template <typename A, typename B>
void BMap<A,B>::save(const char* dir, const char* name)
{
   std::cerr << "save" 
             << " dir " << dir 
             << " name  " << name
             << std::endl ;  
} 


template class BMap<std::string, std::string>;
