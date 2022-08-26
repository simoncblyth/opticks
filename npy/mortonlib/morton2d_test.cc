// ./morton2d_test.sh 

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <bitset>

#include "morton2d.h"

uint64_t mask_(int i){ 
   assert( i > -1 && i < 33 ); 
   uint64_t mask_idx = i ; 
   return  i == 32ull ? 0ull : ~(( 0x1ull << 2ull*i ) - 1ull) ; 
}
std::string desc_mask_()
{
    std::stringstream ss ;
    ss << "desc_mask_" << std::endl ;  
    for(unsigned i=0 ; i < 33 ; i++) ss << std::setw(2) << i << " | " << std::bitset<64>(mask_(i)) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
desc_mask_
 0 | 1111111111111111111111111111111111111111111111111111111111111111
 1 | 1111111111111111111111111111111111111111111111111111111111111100
 2 | 1111111111111111111111111111111111111111111111111111111111110000
 3 | 1111111111111111111111111111111111111111111111111111111111000000
 4 | 1111111111111111111111111111111111111111111111111111111100000000
 5 | 1111111111111111111111111111111111111111111111111111110000000000
 6 | 1111111111111111111111111111111111111111111111111111000000000000
 7 | 1111111111111111111111111111111111111111111111111100000000000000
 8 | 1111111111111111111111111111111111111111111111110000000000000000
 9 | 1111111111111111111111111111111111111111111111000000000000000000
10 | 1111111111111111111111111111111111111111111100000000000000000000
11 | 1111111111111111111111111111111111111111110000000000000000000000
12 | 1111111111111111111111111111111111111111000000000000000000000000
13 | 1111111111111111111111111111111111111100000000000000000000000000
14 | 1111111111111111111111111111111111110000000000000000000000000000
15 | 1111111111111111111111111111111111000000000000000000000000000000
16 | 1111111111111111111111111111111100000000000000000000000000000000
17 | 1111111111111111111111111111110000000000000000000000000000000000
18 | 1111111111111111111111111111000000000000000000000000000000000000
19 | 1111111111111111111111111100000000000000000000000000000000000000
20 | 1111111111111111111111110000000000000000000000000000000000000000
21 | 1111111111111111111111000000000000000000000000000000000000000000
22 | 1111111111111111111100000000000000000000000000000000000000000000
23 | 1111111111111111110000000000000000000000000000000000000000000000
24 | 1111111111111111000000000000000000000000000000000000000000000000
25 | 1111111111111100000000000000000000000000000000000000000000000000
26 | 1111111111110000000000000000000000000000000000000000000000000000
27 | 1111111111000000000000000000000000000000000000000000000000000000
28 | 1111111100000000000000000000000000000000000000000000000000000000
29 | 1111110000000000000000000000000000000000000000000000000000000000
30 | 1111000000000000000000000000000000000000000000000000000000000000
31 | 1100000000000000000000000000000000000000000000000000000000000000
32 | 0000000000000000000000000000000000000000000000000000000000000000
**/


std::string desc(uint64_t x0, uint64_t y0, int imask=0 )
{
    uint64_t mask = mask_(imask) ; 

    morton2d<uint64_t> m0(x0,y0); 
    uint64_t x0_, y0_ ;
    m0.decode( x0_, y0_ ); 
    assert( x0 == x0_ ); 
    assert( y0 == y0_ ); 

    uint64_t s32=0xffffffff ; 
    double X0 = double(x0)/double(s32) ; 
    double Y0 = double(y0)/double(s32) ; 

    const uint64_t& k0 = m0.key ; 
    uint64_t k0_and_mask = k0 & mask ;       // changes block of least significant bits to zeros
    uint64_t k0_or_antimask = k0 | ~mask ;   // changes block of least significant bits to ones 

    morton2d<uint64_t> m1(k0_and_mask); 
    uint64_t x1, y1 ;
    m1.decode( x1, y1 ); 
    double X1 = double(x1)/double(s32) ; 
    double Y1 = double(y1)/double(s32) ; 

    morton2d<uint64_t> m2(k0_or_antimask); 
    uint64_t x2, y2 ;
    m2.decode( x2, y2 ); 
    double X2 = double(x2)/double(s32) ; 
    double Y2 = double(y2)/double(s32) ; 


    std::stringstream ss ; 
    ss 
       << std::setw(12) << " mask " 
       << std::setw(16) << std::hex << mask << std::dec << " | " << std::setw(64) << std::bitset<64>(mask)
       << std::setw(12) << " imask " 
       << std::setw(2) << imask 
       << std::endl
       << std::endl
       << std::setw(12) << " k0 & mask " 
       << std::setw(16) << std::hex << k0_and_mask << std::dec << " | " << std::setw(64) << std::bitset<64>(k0_and_mask)
       << std::endl
       << std::setw(12) << " k0 " 
       << std::setw(16) << std::hex << k0 << std::dec << " | " << std::setw(64) << std::bitset<64>(k0)
       << std::endl
       << std::setw(12) << " k0 | ~mask " 
       << std::setw(16) << std::hex << k0_or_antimask << std::dec << " | " << std::setw(64) << std::bitset<64>(k0_or_antimask)
       << std::endl
       << std::endl
       << std::setw(12) << " x1 " 
       << std::setw(16) << std::hex << x1 << std::dec << " | " << std::setw(64) << std::bitset<32>(x1) 
       << std::setw(10) << std::fixed << std::setprecision(5) << X1 
       << std::setw(10) << std::fixed << std::setprecision(5) << X1 - X0
       << std::endl 
       << std::setw(12) << " x0 " 
       << std::setw(16) << std::hex << x0 << std::dec << " | " << std::setw(64) << std::bitset<32>(x0) 
       << std::setw(10) << std::fixed << std::setprecision(5) << X0 
       << std::endl 
       << std::setw(12) << " x2 " 
       << std::setw(16) << std::hex << x2 << std::dec << " | " << std::setw(64) << std::bitset<32>(x2) 
       << std::setw(10) << std::fixed << std::setprecision(5) << X2 
       << std::setw(10) << std::fixed << std::setprecision(5) << X2 - X0
       << std::endl 
       << std::endl
       << std::setw(12) << " y1 " 
       << std::setw(16) << std::hex << y1 << std::dec << " | " << std::setw(64) << std::bitset<32>(y1)
       << std::setw(10) << std::fixed << std::setprecision(5) << Y1 
       << std::setw(10) << std::fixed << std::setprecision(5) << Y1 - Y0
       << std::endl  
       << std::setw(12) << " y0 " 
       << std::setw(16) << std::hex << y0 << std::dec << " | " << std::setw(64) << std::bitset<32>(y0)
       << std::setw(10) << std::fixed << std::setprecision(5) << Y0 
       << std::endl  
       << std::setw(12) << " y2 " 
       << std::setw(16) << std::hex << y2 << std::dec << " | " << std::setw(64) << std::bitset<32>(y2)
       << std::setw(10) << std::fixed << std::setprecision(5) << Y2
       << std::setw(10) << std::fixed << std::setprecision(5) << Y2 - Y0
       << std::endl  
       ; 
    std::string str = ss.str(); 
    return str ; 
}


void morton_interleave_demo()
{
    // bit-by-bit interleaving x to the more significant slot than y 

    int imask = 28 ; 

    std::cout << desc( 0x00000000, 0x00000000, imask) << std::endl ; 
    std::cout << desc( 0x11111111, 0x11111111, imask) << std::endl ; 
    std::cout << desc( 0x22222222, 0x22222222, imask) << std::endl ; 
    std::cout << desc( 0x33333333, 0x33333333, imask) << std::endl ; 
    std::cout << desc( 0x44444444, 0x44444444, imask) << std::endl ; 
    std::cout << desc( 0x55555555, 0x55555555, imask) << std::endl ; 
    std::cout << desc( 0x66666666, 0x66666666, imask) << std::endl ; 
    std::cout << desc( 0x77777777, 0x77777777, imask) << std::endl ; 
    std::cout << desc( 0x88888888, 0x88888888, imask) << std::endl ; 
    std::cout << desc( 0x99999999, 0x99999999, imask) << std::endl ; 
    std::cout << desc( 0xaaaaaaaa, 0xaaaaaaaa, imask) << std::endl ; 
    std::cout << desc( 0xbbbbbbbb, 0xbbbbbbbb, imask) << std::endl ; 
    std::cout << desc( 0xcccccccc, 0xcccccccc, imask) << std::endl ; 
    std::cout << desc( 0xdddddddd, 0xdddddddd, imask) << std::endl ; 
    std::cout << desc( 0xeeeeeeee, 0xeeeeeeee, imask) << std::endl ; 
    std::cout << desc( 0xffffffff, 0xffffffff, imask) << std::endl ; 
}



void test_morton_mask()
{
    uint64_t x0 = 0x11111111 ; 
    uint64_t y0 = 0xffffffff ; 
    std::cout << desc(x0,y0) << std::endl << std::endl ; 

    morton2d<uint64_t> m2(x0,y0); 
    uint64_t k0 = m2.key ; 

    for(unsigned i=0 ; i < 33 ; i++)
    {
        uint64_t mask = mask_(i) ; 
        morton2d<uint64_t> m(k0 & ~mask); 
                
        uint64_t ix, iy ; 
        m.decode(ix, iy) ; 

        std::cout 
            << std::setw(2) << i 
            //<< " | " 
            //<< std::bitset<64>(~mask) 
            << " | " 
            << std::bitset<64>(k0 & ~mask) 
            << " | " 
            << std::endl
            ;   
    }


}


int main(int argc, char** argv)
{
    //std::cout << desc_mask_() ; 

    morton_interleave_demo(); 
    //test_morton_mask(); 

    return 0 ; 
}
