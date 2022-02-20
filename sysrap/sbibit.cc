// name=sbibit ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <cstdio>
#include "sbibit.h"

int main(int argc, char** argv)
{
    for(unsigned a=0 ; a < 4 ; a++)
    for(unsigned b=0 ; b < 4 ; b++)
    for(unsigned c=0 ; c < 4 ; c++)
    for(unsigned d=0 ; d < 4 ; d++)
    {
         unsigned char packed = sbibit_PACK4(a, b, c, d); 

         unsigned a1 = sbibit_UNPACK4_0(packed) ; 
         unsigned b1 = sbibit_UNPACK4_1(packed) ; 
         unsigned c1 = sbibit_UNPACK4_2(packed) ; 
         unsigned d1 = sbibit_UNPACK4_3(packed) ; 
 
         assert( a == a1 ); 
         assert( b == b1 ); 
         assert( c == c1 ); 
         assert( d == d1 ); 

         printf(" a %d b %d c %d d %d \n", a,b,c,d); 

    }
    return 0 ; 
}
