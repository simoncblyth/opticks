#include <iostream>
#include <cassert>

#include "npy_header.hh"

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ;
    
    unsigned arr_bytes  = 0x12345678 ; 
    unsigned meta_bytes = 0xdeadbeef ; 

    npy_header h0(arr_bytes, meta_bytes) ; 

    assert( sizeof(h0) == 8 ); 
    assert( h0.arr_bytes() == arr_bytes ); 
    assert( h0.meta_bytes() == meta_bytes ); 

    npy_header h1 = npy_header::unpack(h0.data()) ;  

    assert( sizeof(h1) == 8 ); 
    assert( h1.arr_bytes() == arr_bytes );  
    assert( h1.meta_bytes() == meta_bytes );  

    std::cout << "h0 " << h0.desc() << std::endl ; 
    std::cout << "h1 " << h1.desc() << std::endl ; 

    return 0 ; 
}

// gcc npy_header.cc npy_header_test.cc -I. -lstdc++ -o /tmp/npy_header_test && /tmp/npy_header_test
