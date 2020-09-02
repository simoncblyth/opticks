// om-;TEST=SPackTest om-t 

#include <cassert>
#include "SPack.hh"

#include "OPTICKS_LOG.hh"


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






int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_Encode_Decode();  
    test_Encode_Decode_ptr();  

    return 0  ; 
}

// om-;TEST=SPackTest om-t

