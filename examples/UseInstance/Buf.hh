#pragma once

#include "USEINSTANCE_API_EXPORT.hh"

struct USEINSTANCE_API Buf
{
    int id ; 
    unsigned num_bytes ;
    void* ptr ;
    Buf(unsigned num_bytes_, void* ptr_) ;
};



