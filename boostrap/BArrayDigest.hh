#pragma once
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

template <typename T>
class BRAP_API BArrayDigest 
{
    public:
       static std::string arraydigest( T* data, unsigned int n );

};
 


