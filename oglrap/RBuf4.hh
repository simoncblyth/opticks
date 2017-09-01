#pragma once

#include <string>
#include "OGLRAP_API_EXPORT.hh"

struct RBuf ; 

struct OGLRAP_API RBuf4
{
    RBuf* x ;
    RBuf* y ;
    RBuf* z ;
    RBuf* w ;

    RBuf* devnull ;


    RBuf4();

    RBuf* at(unsigned i) const ;
    unsigned num_buf() const ;

    std::string desc() const ;


};
 



