#include <cassert>
#include "SPack.hh"

struct C4 
{
    unsigned char x, y, z, w ; 
}; 

union ucccc_t 
{
    unsigned int u ; 
    C4          c4 ; 
};


unsigned SPack::Encode(unsigned char x, unsigned char y, unsigned char z, unsigned char w)  // static 
{
    assert( sizeof(unsigned char) == 1); 
    assert( sizeof(unsigned int) == 4); 

    ucccc_t uc ; 
    uc.c4.x = x ; 
    uc.c4.y = y ; 
    uc.c4.z = z ; 
    uc.c4.w = w ; 

    unsigned int value = uc.u ;   
    return value  ; 
}
unsigned SPack::Encode(const unsigned char* ptr, const unsigned num) // static
{
    assert( num == 4 ); 
    unsigned char x = *(ptr+0) ; 
    unsigned char y = *(ptr+1) ; 
    unsigned char z = *(ptr+2) ; 
    unsigned char w = *(ptr+3) ; 
    return SPack::Encode( x, y, z, w ); 
}


void SPack::Decode( const unsigned int value,  unsigned char& x, unsigned char& y, unsigned char& z, unsigned char& w ) // static
{
    assert( sizeof(unsigned char) == 1); 
    assert( sizeof(unsigned int) == 4); 

    ucccc_t uc ; 
    uc.u = value ; 
    x = uc.c4.x ; 
    y = uc.c4.y ; 
    z = uc.c4.z ; 
    w = uc.c4.w ; 
}

void SPack::Decode( const unsigned int value,  unsigned char* ptr, const unsigned num ) // static
{
    assert( num == 4); 
    assert( sizeof(unsigned char) == 1); 
    assert( sizeof(unsigned int) == 4); 

    ucccc_t uc ; 
    uc.u = value ; 
     
    *(ptr + 0) = uc.c4.x ; 
    *(ptr + 1) = uc.c4.y ; 
    *(ptr + 2) = uc.c4.z ; 
    *(ptr + 3) = uc.c4.w ; 
}


