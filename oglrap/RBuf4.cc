#include <sstream>
#include <cstddef>
#include "RBuf.hh"
#include "RBuf4.hh"

RBuf4::RBuf4()
    :
    x(NULL),
    y(NULL),
    z(NULL),
    w(NULL),
    devnull(NULL)
{
}

RBuf* RBuf4::at(unsigned i) const 
{
    RBuf* b = NULL ; 
    switch(i)
    {
        case 0: b = x ; break ; 
        case 1: b = y ; break ; 
        case 2: b = z ; break ; 
        case 3: b = w ; break ; 
    }  
    return b ; 
}

unsigned RBuf4::num_buf() const 
{
    return ( x ? 1 : 0 ) + ( y ? 1 : 0 ) + ( z ? 1 : 0 ) + ( w ? 1 : 0 ) ;
}


std::string RBuf4::desc() const 
{
    std::stringstream ss ; 

    ss << "RBuf4"  
       << " num_buf " << num_buf()
       << " x " << ( x ? x->brief() : "-" )
       << " y " << ( y ? y->brief() : "-" )
       << " z " << ( z ? z->brief() : "-" )
       << " w " << ( w ? w->brief() : "-" )
       << " devnull " << ( devnull ? devnull->brief() : "-" )
       ;

    return ss.str();
}

