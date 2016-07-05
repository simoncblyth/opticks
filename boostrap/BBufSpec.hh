#pragma once

#include <cstdio>
#include "BRAP_API_EXPORT.hh"

struct BRAP_API BBufSpec {

    int id ; 
    void* ptr ;
    unsigned int num_bytes ; 
    int target ; 

    BBufSpec(int id_, void* ptr_, unsigned int num_bytes_, int target_)
       :
          id(id_),
          ptr(ptr_),
          num_bytes(num_bytes_),
          target(target_)
    {
    }
    void Summary(const char* msg)
    {
        printf("%s : id %d ptr %p num_bytes %d target %d \n", msg, id, ptr, num_bytes, target ); 
    }

};

#include "BRAP_TAIL.hh"

