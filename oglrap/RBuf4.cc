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



RBuf4* RBuf4::MakeFork(const RBuf* src, int num, int debug_clone_slot )
{
    assert( num < 4 && num > 0 );
    RBuf4* fork = new RBuf4 ; 

    for(int i=0 ; i < num ; i++)
    {
        RBuf* b = NULL ; 

        if( i == debug_clone_slot )
        {
             b = src->clone() ;
             b->gpu_resident = false ;    // non-GPU residents need to be actually uploaded from host  
        }
        else
        {
             b = src->cloneZero() ;
             b->gpu_resident = true ;   // GPU residents are not uploaded, CPU side is a monicker
        }
        fork->set(i, b );

    }
    fork->devnull = new RBuf(0,1,0,NULL);  // 1-byte buffer used with workaround

    return fork ; 
}





