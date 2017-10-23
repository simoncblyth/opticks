#pragma once

#include "NPY_API_EXPORT.hh"
 
typedef enum 
{
   NOCT_ZERO, 
   NOCT_INTERNAL, 
   NOCT_LEAF 
}           NOctNode_t ;

struct NPY_API NOctNodeEnum
{
    static const char* NOCT_ZERO_     ;
    static const char* NOCT_INTERNAL_ ;
    static const char* NOCT_LEAF_     ;

    static const char* NOCTName(NOctNode_t type);
};



