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
            << " ulo "    << std::setw(8) << ulo() 
            << " uhi "    << std::setw(8) << uhi() 
            << "    "
            << std::dec
            << " s2.x " << std::setw(6) << s2.x 
            << " s2.y " << std::setw(6) << s2.y
            << " u "    << std::setw(12) << u 
            << " ulo "    << std::setw(8) << ulo() 
            << " uhi "    << std::setw(8) << uhi() 
            << std::endl
            ; 
    }
};


void test_ctor(int x, int y, bool dump)
{
    s2u e ;  // construct the 32 bit unsigned from two assumed 16 bit signed 
    e.u =  (( y << 16 ) & 0xffff0000 ) | (( x & 0x0000ffff ) >> 0 );
    if(dump) e.dump("e"); 

    s2u a ;   // construct using the short2 uniform 
    a.s2.x = x ; 
    a.s2.y = y ; 
    if(dump) a.dump("a"); 
    assert( a.ulo() == a.s2.x ); // least significant two bytes of the unsigned align with the s2.x due to little endianness
    assert( a.uhi() == a.s2.y );  

    s2u b ;   // construct using lo and hi from a  
    b.s2.x = a.ulo() ; 
    b.s2.y = a.uhi() ; 
    if(dump) b.dump("b"); 
    assert( a.u == b.u ); 

    s2u c ;  // directly from a.u 
    c.u = a.u ; 
    if(dump) c.dump("c"); 
    assert( c.s2.x == x );  
    assert( c.s2.y == y );  

    s2u d(a);   // default copy ctor
    if(dump) d.dump("d"); 
    assert( d.u == a.u ); 
}


void test_yflip(int x, int y, bool dump)
{
    s2u a ; 
    a.s2.x = x ; 
    a.s2.y = y ; 

    s2u cy0 ; 
    cy0.s2.x = x ; 
    cy0.s2.y = -y ; 
    if(dump) cy0.dump("cy0"); 

    s2u cy1(a) ; 
    cy1.yflip() ; 
    if(dump) cy1.dump("cy1"); 
    assert( cy0.u == cy1.u ); 

    s2u cy2(a);
    cy2.u = (( ~( cy2.u >> 16 ) + 1 ) << 16 ) | ( 0x0000ffff & cy2.u )  ; 
    // inversion and adding one is same as changing sign in twos complement 
    // have to bring the hi bytes down to ground in order to add one and then scoot them up again

    //cy2.u = (( -( cy2.u >> 16 ) ) << 16 )  | ( 0x0000ffff & cy2.u )  ; 


    if(dump) cy2.dump("cy2"); 
    assert( cy0.u == cy2.u );  
}


void test_xflip(int x, int y, bool dump)
{
    s2u a ; 
    a.s2.x = x ; 
    a.s2.y = y ; 

    s2u cx0 ; 
    cx0.s2.x = -x ; 
    cx0.s2.y = y ; 
    if(dump) cx0.dump("cx0"); 

    s2u cx1(a) ; 
    cx1.xflip() ; 
    if(dump) cx1.dump("cx1"); 
    assert( cx0.u == cx1.u ); 

    s2u cx2(a) ; 
    cx2.u =  ( cx2.u & 0xffff0000 ) | ((~( 0x0000ffff & cx2.u ) + 1) & 0x0000ffff ) ; 
    if(dump) cx2.dump("cx2"); 
    assert( cx0.u == cx2.u ); 
}

void test_full()
{
    bool dump = false ; 
    int x0 = -0x8000 ; 
    int x1 =  0x7fff ; 
    int xs = 10 ; 
    int y0 = -0x8000 ; 
    int y1 =  0x7fff ; 
    int ys = 10 ; 

    for(int x=x0 ; x <= x1 ; x+=xs) for(int y=y0 ; y <= y1 ; y+=ys)
    {
        dump = abs(y) <= 10 && abs(y) <= 10 ;  
        test_ctor(x,y,dump); 
        test_xflip(x,y,dump); 
        test_yflip(x,y,dump); 
    } 
}

void test_small()
{
    bool dump = true ; 
    int x0 =  0 ; 
    int x1 =  100 ; 
    int xs =  1 ; 
    int y0 =  0 ; 
    int y1 =  100 ; 
    int ys =  1 ; 

    for(int x=x0 ; x <= x1 ; x+=xs) for(int y=y0 ; y <= y1 ; y+=ys)
    {
        test_ctor(x,y,dump); 
        test_xflip(x,y,dump); 
        test_yflip(x,y,dump); 
    } 
}


int main(int argc, char** argv)
{
    test_full(); 
    //test_small(); 

    return 0 ; 
}
