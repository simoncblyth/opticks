// ./morton2d_test.sh 

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <bitset>

#include "morton2d.h"


void morton_interleave_demo(uint64_t x , uint64_t y, const char* msg)
{
    morton2d<uint64_t> m(x,y); 
    const uint64_t& k = m.key ; 

    uint64_t x2, y2 ;
    m.decode( x2, y2 ); 

    assert( x == x2 ); 
    assert( y == y2 ); 

    std::bitset<32> sx(x) ; 
    std::bitset<32> sy(y) ; 
    std::bitset<64> sk(k) ; 

    std::cout << "morton_interleave_demo : " << msg << std::endl ; 
    std::cout << " x " << std::setw(16) << std::hex << x << std::dec << "|" << std::setw(64) << sx << std::endl ; 
    std::cout << " y " << std::setw(16) << std::hex << y << std::dec << "|" << std::setw(64) << sy << std::endl ; 
    std::cout << " k " << std::setw(16) << std::hex << k << std::dec << "|" << std::setw(64) << sk << std::endl ; 
}

void morton_interleave_demo()
{
    morton_interleave_demo( 0x00000000, 0xffffffff, "bit-by-bit interleaving x to the more significant slot than y "); 
    morton_interleave_demo( 0xffffffff, 0x00000000, "bit-by-bit interleaving x to the more significant slot than y "); 
    morton_interleave_demo( 0x77777777, 0x77777777, "bit-by-bit interleaving x to the more significant slot than y "); 
}


int main(int argc, char** argv)
{
    morton_interleave_demo(); 

    return 0 ; 
}
