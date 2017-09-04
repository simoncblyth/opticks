#pragma once

#include <string>
#include "OGLRAP_API_EXPORT.hh"

struct RBuf ; 

struct OGLRAP_API RBuf4
{
    static RBuf4* MakeFork(const RBuf* src, int num, int debug_clone_slot=-1 );
    static RBuf4* MakeDevNull(unsigned num_buf, unsigned num_bytes );

    RBuf* x ;
    RBuf* y ;
    RBuf* z ;
    RBuf* w ;

    RBuf4();

    RBuf* at(unsigned i) const ;
    void  set(unsigned i, RBuf* b) ;
    unsigned num_buf() const ;

    std::string desc() const ;
    void dump() const ;


    void uploadNull(GLenum target, GLenum usage );
 
    void pullback(const char* msg="RBuf4::pullback");
    void bind();


};
 



