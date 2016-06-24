#pragma once

#include <cstdio>
#include "BRAP_API_EXPORT.hh"

struct BRAP_API BBufSpec {

    int id ; 
    void* ptr ;
    unsigned int num_bytes ; 
    int target ; 

    BBufSpec(int id, void* ptr, unsigned int num_bytes, int target)
       :
          id(id),
          ptr(ptr),
          num_bytes(num_bytes),
          target(target)
    {
    }
    void Summary(const char* msg)
    {
        printf("%s : id %d ptr %p num_bytes %d target %d \n", msg, id, ptr, num_bytes, target ); 
    }

};
