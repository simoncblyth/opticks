#include <cassert>
#include "SPack.hh"

struct C4 
{
    unsigned char x, y, z, w ; 
}; 

struct u2
{
    unsigned x, y  ; 
}; 

union ucccc_t 
{
    unsigned int u ; 
    C4          c4 ; 
};

bool SPack::IsLittleEndian()
{
    int n = 1; 
    return (*(char *)&n == 1) ;
}


unsigned SPack::Encode(unsigned x, unsigned y, unsigned z, unsigned w)  // static 
{
    assert( x <= 0xff ); 
    assert( y <= 0xff ); 
    assert( z <= 0xff ); 
    assert( w <= 0xff ); 

    unsigned char xc = x ; 
    unsigned char yc = y ; 
    unsigned char zc = z ; 
    unsigned char wc = w ; 

    return SPack::Encode(xc, yc, zc, wc); 
}


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
    unsigned packed = ( a << 16 ) | ( b << 0 ) ; 
    return packed  ; 
}
void SPack::Decode22( const unsigned packed, unsigned& hi, unsigned& lo ) // static
{
    assert( sizeof(unsigned) == 4 ); 
    hi = ( packed & 0xffff0000 ) >> 16 ;  
    lo = ( packed & 0x0000ffff ) >>  0 ;
}

unsigned SPack::Decode22a( const unsigned packed ) // static
{
    return ( packed & 0xffff0000 ) >> 16  ; 
}
unsigned SPack::Decode22b( const unsigned packed ) // static
{
    return ( packed & 0x0000ffff ) >>  0 ; 
}

/**
SPack::Decode22hilo SPack::Decode22hi SPack::Decode22lo
-----------------------------------------------------------

Signed variants of SPack::Decode22 SPack::Decode22a and SPack::Decode22b

**/

unsigned SPack::Encode22hilo( int a, int b )
{
    assert( a >= -0x8000 && a <= 0x7fff );   // 16 bit signed range     0x8000 - 0x10000 = -0x8000
    assert( b >= -0x8000 && b <= 0x7fff ); 
    unsigned packed = (( a & 0x0000ffff ) << 16 ) | (( b & 0x0000ffff ) << 0 ) ;  
    return packed ; 
}

void SPack::Decode22hilo( const unsigned packed, int& a, int& b ) // static
{
    unsigned hi = ( packed & 0xffff0000 ) >> 16 ;
    unsigned lo = ( packed & 0x0000ffff ) >>  0 ;

    a = hi <= 0x7fff ? hi : hi - 0x10000 ;    // 16-bit twos complement
    b = lo <= 0x7fff ? lo : lo - 0x10000 ;    
}

int SPack::Decode22hi( const unsigned packed ) // static
{
    unsigned hi = ( packed & 0xffff0000 ) >> 16 ;
    return hi <= 0x7fff ? hi : hi - 0x10000 ;    // 16-bit twos complement
}
int SPack::Decode22lo( const unsigned packed ) // static
{
    unsigned lo = ( packed & 0x0000ffff ) >>  0 ;
    return lo <= 0x7fff ? lo : lo - 0x10000 ;    // 16-bit twos complement
}




void SPack::Decode( const unsigned int value,  unsigned& x, unsigned& y, unsigned& z, unsigned& w ) // static
{
    unsigned char ucx ; 
    unsigned char ucy ; 
    unsigned char ucz ; 
    unsigned char ucw ; 

    Decode(value, ucx, ucy, ucz, ucw ); 

    x = ucx ; 
    y = ucy ; 
    z = ucz ; 
    w = ucw ; 
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


float SPack::unsigned_as_float( const unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}

double SPack::unsigned_as_double( const unsigned x, const unsigned y ) 
{
    union { u2 uu ; double d ; } uud ;   
    uud.uu.x = x  ;   
    uud.uu.y = y  ;   
    return uud.d ; 
}

void SPack::double_as_unsigned(unsigned& x, unsigned& y, const double d ) 
{
    union { u2 uu ; double d ; } uud ;   
    uud.d = d ; 
    x = uud.uu.x ;
    y = uud.uu.y ;   
}




unsigned SPack::float_as_unsigned( const float f ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.f = f  ;   
    return uif.u ; 
}






/**
SPack::unsigned_as_int
-----------------------

The bits of unsigned integers can hold the bits of a signed int without problem 
(within the signed range), thus can reinterpret those bits as a signed integer 
using twos-complement.  Notice how the number of bits is relevant to the bit field 
representation of negative integers in a way that is not the case for positive ones.

**/

template <int NUM_BITS>
int SPack::unsigned_as_int(unsigned value)   // static
{  
    unsigned twos_complement_sum = 0x1 << NUM_BITS ;   
    unsigned signed_max =  (0x1 << (NUM_BITS-1)) - 1  ; 
    int ivalue = value <= signed_max ? value : value - twos_complement_sum ; 
    return ivalue ; 
}



template int SPack::unsigned_as_int<8>(unsigned value) ;
template int SPack::unsigned_as_int<16>(unsigned value) ;


int SPack::unsigned_as_int_32(unsigned value)
{
    uif_t uif ; 
    uif.u = value ; 
    return uif.i ; 
}
int SPack::unsigned_as_int_16(unsigned value)
{
    ui16_t ui ; 
    ui.u = value ; 
    return ui.i ; 
}





