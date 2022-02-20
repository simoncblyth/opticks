// name=sbit_ ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <cstdio>
#include "sbit_.h"

int main(int argc, char** argv)
{
    for(unsigned a=0 ; a < 2 ; a++)
    for(unsigned b=0 ; b < 2 ; b++)
    for(unsigned c=0 ; c < 2 ; c++)
    for(unsigned d=0 ; d < 2 ; d++)
    for(unsigned e=0 ; e < 2 ; e++)
    for(unsigned f=0 ; f < 2 ; f++)
    for(unsigned g=0 ; g < 2 ; g++)
    for(unsigned h=0 ; h < 2 ; h++)
    {
         unsigned char packed0 = sbit_rPACK8(a, b, c, d, e, f, g, h); 

         bool _a(a) ; 
         bool _b(b) ; 
         bool _c(c) ; 
         bool _d(d) ; 
         bool _e(e) ; 
         bool _f(f) ; 
         bool _g(g) ; 
         bool _h(h) ; 

         unsigned char packed1 = sbit_rPACK8(_a, _b, _c, _d, _e, _f, _g, _h); 
         assert( packed0 == packed1 ); 
         unsigned char packed = packed0 ; 

         unsigned a1 = sbit_rUNPACK8_0(packed) ; 
         unsigned b1 = sbit_rUNPACK8_1(packed) ; 
         unsigned c1 = sbit_rUNPACK8_2(packed) ; 
         unsigned d1 = sbit_rUNPACK8_3(packed) ; 
         unsigned e1 = sbit_rUNPACK8_4(packed) ; 
         unsigned f1 = sbit_rUNPACK8_5(packed) ; 
         unsigned g1 = sbit_rUNPACK8_6(packed) ; 
         unsigned h1 = sbit_rUNPACK8_7(packed) ; 
 
         assert( a == a1 ); 
         assert( b == b1 ); 
         assert( c == c1 ); 
         assert( d == d1 ); 
         assert( e == e1 ); 
         assert( f == f1 ); 
         assert( g == g1 ); 
         assert( h == h1 ); 

         printf(" a %d b %d c %d d %d e %d f %d g %d h %d    %d%d%d%d%d%d%d%d    %4d  \n", a,b,c,d,e,f,g,h, a,b,c,d,e,f,g,h, packed ); 
    }
    return 0 ; 
}
