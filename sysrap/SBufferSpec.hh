#pragma once

#include <cstdlib>
#include <string>
#include "SYSRAP_API_EXPORT.hh"

// probably you need NBufferSpec

struct SYSRAP_API SBufferSpec 
{  
    std::size_t bufferByteLength ; 
    std::size_t headerByteLength ; 
    std::string uri ; 
    void*       ptr ; 

    std::size_t dataSize() const 
    {   
        return bufferByteLength - headerByteLength ; 
    }   
 
};

