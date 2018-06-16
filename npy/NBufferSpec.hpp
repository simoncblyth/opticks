#pragma once

#include <cstdlib>
#include <string>
#include "NPY_API_EXPORT.hh"

// Cannot dynamic_cast up from void* as no vtable 
// so use an NPYBase* ptr to allow collections of 
// various types of NPY buffers

class NPYBase ; 


struct NPY_API NBufferSpec 
{  
    std::size_t bufferByteLength ; 
    std::size_t headerByteLength ; 
    std::string uri ; 
    const NPYBase*  ptr ; 

    std::size_t dataSize() const 
    {   
        return bufferByteLength - headerByteLength ; 
    }   
 
};


