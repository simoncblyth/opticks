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


unsigned SPack::Encode13(unsigned char c, unsigned int ccc)  // static 
{
    assert( (ccc & (0xff << 24)) == 0 ); 
    unsigned int value = ccc | ( c << 24 ) ; 
    return value  ; 
}

void SPack::Decode13( const unsigned int value, unsigned char& c, unsigned int& ccc ) // static
{
    c = ( value >> 24 ) & 0xff ;  
    ccc = value & 0xffffff ; 
}

unsigned SPack::Encode22(unsigned a, unsigned b)  // static 
{
    assert( sizeof(unsigned) == 4 ); 
    assert( (a & 0xffff0000) == 0 ); 
    assert( (b & 0xffff0000) == 0 ); 
    unsigned value = ( a << 16 ) | ( b << 0 ) ; 
    return value  ; 
}

void SPack::Decode22( const unsigned value, unsigned& a, unsigned& b ) // static
{
    assert( sizeof(unsigned) == 4 ); 
    a = ( value >> 16 ) & 0xffff ;  
    b = ( value >>  0 ) & 0xffff ; 
}

unsigned SPack::Decode22a( const unsigned value ) // static
{
    return ( value >> 16 ) & 0xffff ; 
}
unsigned SPack::Decode22b( const unsigned value ) // static
{
    return ( value >>  0 ) & 0xffff ; 
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



float SPack::int_as_float(const int i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}
int SPack::int_from_float(const float f)
{
    uif_t uif ; 
    uif.f = f ; 
    return uif.i ; 
}
float SPack::uint_as_float(const unsigned i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}
unsigned SPack::uint_from_float(const float f)
{
    uif_t uif ; 
    uif.f = f ; 
    return uif.u ; 
}



