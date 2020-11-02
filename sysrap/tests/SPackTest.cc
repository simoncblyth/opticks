// om-;TEST=SPackTest om-t 

#include <cassert>
#include "SPack.hh"
#include "OPTICKS_LOG.hh"

void test_Encode_Decode_unsigned()
{
    unsigned x = 0x00 ;   
    unsigned y = 0xaa ;   
    unsigned z = 0xbb ;   
    unsigned w = 0xff ;

    unsigned value_expect = 0xffbbaa00 ; 
    unsigned value = SPack::Encode(x,y,z,w); 

    LOG(info) 
        << " value " << std::hex << value 
        << " value_expect " << std::hex << value_expect
        ;

    assert( value == value_expect ); 

    unsigned x_ ; 
    unsigned y_ ; 
    unsigned z_ ; 
    unsigned w_ ; 

    SPack::Decode( value, x_, y_, z_, w_ ); 

    assert( x == x_ );   
    assert( y == y_ );   
    assert( z == z_ );   
    assert( w == w_ );   
}


void test_Encode()
{
    unsigned char nu = 10 ; 
    unsigned char nv = 10 ; 
    unsigned char nw =  4 ; 

    unsigned char u = nu - 1 ; 
    unsigned char v = nv - 1  ; 
    unsigned char w = nw - 1 ; 

    unsigned int packed = SPack::Encode(u,v,w,0); 
    LOG(info) 
        << " u " << u 
        << " v " << v 
        << " w " << w 
        << " packed " << packed 
        ;

}


void test_Encode_Decode()
{
    unsigned char x = 1 ; 
    unsigned char y = 128 ; 
    unsigned char z = 255 ; 
    unsigned char w = 128 ; 

    unsigned int value = SPack::Encode(x,y,z,w); 
    LOG(info) << " value " << value  ; 

    unsigned char x2, y2, z2, w2 ; 
    SPack::Decode( value, x2, y2, z2, w2 ); 

    assert( x == x2 ); 
    assert( y == y2 ); 
    assert( z == z2 ); 
    assert( w == w2 ); 
}


void test_Encode_Decode_ptr()
{
    unsigned char a[4] ; 
    a[0] = 1 ; 
    a[1] = 128 ; 
    a[2] = 255 ; 
    a[3] = 128 ; 

    unsigned int value = SPack::Encode(a, 4); 
    LOG(info) << " value " << value  ; 

    unsigned char b[4] ; 
    SPack::Decode( value, b, 4 ); 

    assert( a[0] == b[0] ); 
    assert( a[1] == b[1] ); 
    assert( a[2] == b[2] ); 
    assert( a[3] == b[3] ); 
}


void test_Encode13_Decode13()
{
    LOG(info); 

    unsigned char c  = 0xff ; 
    unsigned int ccc   = 0xffffff ; 
    unsigned expect  = 0xffffffff ; 

    unsigned value = SPack::Encode13( c, ccc );  
    assert( value == expect ); 

    unsigned char c2 ; 
    unsigned int  ccc2 ; 
    SPack::Decode13( value, c2, ccc2 ); 
    assert( c == c2 ); 
    assert( ccc == ccc2 ); 
}


void test_Encode22_Decode22()
{
    LOG(info); 

    unsigned a0 = 0xdead ; 
    unsigned b0 = 0xbeef ; 
    unsigned expect  = 0xdeadbeef ; 

    unsigned value = SPack::Encode22( a0, b0 );  
    assert( value == expect ); 

    unsigned a1 ; 
    unsigned b1 ; 
    SPack::Decode22( value, a1, b1 ); 
    assert( a0 == a1 ); 
    assert( b0 == b1 ); 

    unsigned a2 = SPack::Decode22a( value ); 
    unsigned b2 = SPack::Decode22b( value ); 
    assert( a0 == a2 ); 
    assert( b0 == b2 ); 

}





void test_int_as_float()
{
    int i0 = -420042 ;  
    float f0 = SPack::int_as_float( i0 ); 
    int i1 = SPack::int_from_float( f0 ); 
    assert( i0 == i1 ); 
    LOG(info) << " i0 " << i0 << " f0 " << f0 << " i1 " << i1 << " (NaN is expected) " ; 
}

void test_uint_as_float()
{
    unsigned u0 = 420042 ;  
    float f0 = SPack::uint_as_float( u0 ); 
    unsigned u1 = SPack::uint_from_float( f0 ); 
    assert( u0 == u1 ); 
    LOG(info) << " u0 " << u0 << " f0 " << f0 << " u1 " << u1 ; 
}


void test_IsLittleEndian()
{
    const char* LE = "LITTLE_ENDIAN : least significant byte at smaller memory address " ; 
    const char* BE = "BIG_ENDIAN    : most significant byte at smaller memory address " ; 
    LOG(info) << ( SPack::IsLittleEndian() ? LE : BE  ) ; 
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    //test_Encode();  

    //test_Encode_Decode();  
    //test_Encode_Decode_ptr();  
    //test_Encode13_Decode13();  
    //test_Encode22_Decode22();  

    //test_int_as_float(); 
    //test_uint_as_float(); 

    //test_Encode_Decode_unsigned();  
    test_IsLittleEndian();  

    return 0  ; 
}

// om-;TEST=SPackTest om-t

