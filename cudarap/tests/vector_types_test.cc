// nvcc vector_types_test.cc -lstdc++ -o /tmp/vector_types_test && /tmp/vector_types_test
#include <cassert>
#include <iostream>
#include <iomanip>

#include <vector_types.h>

union s2u
{
    short2   s2 ;  
    unsigned  u ; 

    void xflip(){ s2.x = -s2.x ; }  // lo
    void yflip(){ s2.y = -s2.y ; }  // hi

    short ulo(){ return ( 0x0000ffff & u ) >>  0 ; }
    short uhi(){ return ( 0xffff0000 & u ) >> 16 ; }

    void dump(const char* label)
    {
        std::cout 
            << std::setw(5) << label 
            << std::hex
            << " s2.x " << std::setw(4) << s2.x 
            << " s2.y " << std::setw(4) << s2.y
            << " u "    << std::setw(8) << u 
            << "    "
            << std::dec
            << " s2.x " << std::setw(6) << s2.x 
            << " s2.y " << std::setw(6) << s2.y
            << " u "    << std::setw(12) << u 
            << std::endl
            ; 
    }
};

int main(int argc, char** argv)
{
    int x =  0x7fff ; 
    int y = -0x8000 ;  

    s2u a ; 
    a.s2.x = x ; 
    a.s2.y = y ; 
    a.dump("a"); 
    assert( a.ulo() == a.s2.x ); // least significant two bytes of the unsigned align with the s2.x due to little endianness
    assert( a.uhi() == a.s2.y );  

    s2u b ; 
    b.s2.x = a.ulo() ; 
    b.s2.y = a.uhi() ; 
    b.dump("b"); 
    assert( a.u == b.u ); 

    s2u c,cx,cy ; 
    c.u = a.u ; 
    c.dump("c"); 

    cx.u = a.u ; cx.xflip() ;  
    cx.dump("cx"); 

    cy.u = a.u ; cy.yflip() ; 
    cy.dump("cy"); 

    s2u d(a);   // default copy ctor
    d.dump("d"); 
    assert( d.u == a.u ); 

    return 0 ; 
}
