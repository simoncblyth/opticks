//https://stackoverflow.com/questions/2437283/c-c-packing-signed-char-into-int

#include <stdint.h>
#include <iostream>
#include <iomanip>

// bits of signed integers can be carried within unsigned ones without problem
uint32_t pack_helper(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
    return c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
}

uint32_t pack(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
    return pack_helper(c0, c1, c2, c3);
}

template <int N>
uint8_t unpack_u(uint32_t packed) {   // cast to avoid potential warnings for implicit narrowing conversion
    return static_cast<uint8_t>(packed >> (N*8));
}

template <int N>
int8_t unpack_s(uint32_t packed) {
    uint8_t r = unpack_u<N>(packed);
    return (r <= 0x7f ? r : r - 0x100 );   // b0111 = x7,   0x7f = 127, 0x100 = 256 
}

/**
https://en.wikipedia.org/wiki/Two%27s_complement

The two's complement of an N-bit number is defined as its complement with respect to 2^N = 0x1 << N ; (one bit past the end) 
the sum of a number and its two's complement is 2^N = 0x1 << N 

**/


int main() {

    int x0 = 254 ; 
    int y0 = 127 ; 
    int z0 = -128 ; 
    int w0 = -7 ; 

    uint32_t pk = pack(x0,y0,z0,w0);

    int x1u = (int)unpack_u<0>(pk) ; 
    int y1u = (int)unpack_u<1>(pk) ; 
    int z1u = (int)unpack_u<2>(pk) ; 
    int w1u = (int)unpack_u<3>(pk) ; 

    int x1s = (int)unpack_s<0>(pk) ; 
    int y1s = (int)unpack_s<1>(pk) ; 
    int z1s = (int)unpack_s<2>(pk) ; 
    int w1s = (int)unpack_s<3>(pk) ; 


    std::cout 
        << std::setw(10) << x0 
        << std::setw(10) << x1u
        << std::setw(10) << x1s
        << std::endl 
        << std::setw(10) << y0 
        << std::setw(10) << y1u
        << std::setw(10) << y1s
        << std::endl 
        << std::setw(10) << z0 
        << std::setw(10) << z1u
        << std::setw(10) << z1s
        << std::endl 
        << std::setw(10) << w0 
        << std::setw(10) << w1u
        << std::setw(10) << w1s
        << std::endl 
        ;

    return 0 ; 
}

// clang SignedIntPackTest.cc -lc++ -o /tmp/SignedIntPackTest &&  /tmp/SignedIntPackTest
