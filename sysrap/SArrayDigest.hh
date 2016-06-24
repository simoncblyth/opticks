#pragma once
#include <string>

#include "SYSRAP_API_EXPORT.hh"

template <typename T>
class SYSRAP_API SArrayDigest 
{
    public:
       static std::string arraydigest( T* data, unsigned int n );

};
 


