// name=uint128 ; nvcc $name.cu -std=c++11 -lstdc++ -ccbin /usr/bin/clang -o /tmp/$name && /tmp/$name

#include <cassert>
#include <stdio.h>
#include "uint128.h"

void print( uint4& u, unsigned s, signed char c, signed char c2  )
{
    printf(" s:%2d c:%3d c2:%3d u:(%8x %8x %8x %8x)\n", s, c, c2, u.x, u.y, u.z, u.w); 
}

int main()
{
    uint4 u = {0,0,0,0} ; 
    for(unsigned s=0 ; s < 16 ; s++)
    {
        int i = -100 - int(s) ; 

        signed char c = i ; 
        signed char c2 ;  
        uint128_setbyte(u, c, s ); 
        uint128_getbyte(u, c2, s ); 

        print(u, s, c, c2 ); 
        assert( c == c2 ); 
    }



    uint4c16 uc0 = {0,0,0,0} ;  
    for(unsigned s=0 ; s < 16 ; s++)
    {
        int i = -100 - int(s) ; 


        signed char c = i ; 
        uc0.c[s] = c ;     
        signed char c2 = uc0.c[s] ; 
     
        print(uc0.u, s, c, c2); 
        assert( c == c2 ); 
    }




    uint4c16 uc1 = {0,0,0,0} ;  
    for(unsigned s=0 ; s < 16 ; s++)
    {
        int i = -100 - int(s) ; 

        signed char c = i ; 

        uc1.c[s] = c ;     
        signed char c2 = uc1.c[s] ; 

        int i2 = c2 ; 
     
        print(uc1.u, s, c, c2); 
        assert( c == c2 ); 
        assert( i == i2 ); 
    }



    return 0 ; 
}
