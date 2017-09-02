#pragma once

#include <string>
#include "OGLRAP_API_EXPORT.hh"

struct RBuf ; 

struct OGLRAP_API RBuf4
{
    static RBuf4* MakeFork(const RBuf* src, unsigned num);

    RBuf* x ;
    RBuf* y ;
    RBuf* z ;
    RBuf* w ;
    RBuf* devnull ;

    RBuf4();

    RBuf* at(unsigned i) const ;
    void  set(unsigned i, RBuf* b) ;
    unsigned num_buf() const ;

    std::string desc() const ;


};
 



