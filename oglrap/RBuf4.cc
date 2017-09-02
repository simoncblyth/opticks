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


void RBuf4::set(unsigned i, RBuf* b) 
{
    switch(i)
    {
        case 0: x = b ; break ; 
        case 1: y = b ; break ; 
        case 2: z = b ; break ; 
        case 3: w = b ; break ; 
    }  
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



RBuf4* RBuf4::MakeFork(const RBuf* src, unsigned num)
{
    assert( num < 4 && num > 0 );
    RBuf4* fork = new RBuf4 ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        RBuf* b = src->cloneZero() ;
        fork->set(i, b );
        b->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY) ;
    }
    fork->devnull = new RBuf(0,1,0,NULL);  // 1-byte buffer used with workaround
    fork->devnull->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY)  ;

    return fork ; 
}





