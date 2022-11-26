// ./domain2d_test.sh 

#include <cstdlib>
#include <bitset>

#include "domain2d.h"

const char* FOLD = getenv("FOLD"); 
const int NBIT = std::atoi( getenv("NBIT") ? getenv("NBIT") : "32" ); 


template <typename T>
static constexpr T sbitmask(unsigned n)
{
    return n == 0 ? 0 : ( (~static_cast<T>(0)) >> ((8*sizeof(T)) - n));
}



void morton_circle_demo()
{
    domain2d dom( -100, 100, -100, 100 ); 
    std::vector<uint64_t> kk ;
    dom.get_circle(kk, 50.f ); 

    NP* b = nullptr ; 

    for(int nbit=0 ; nbit <=64 ; nbit++)
    {
        uint64_t mask = ~sbitmask<uint64_t>(64-nbit) ; 
        NP* a = dom.make_array( kk, mask ); 
        if( nbit == NBIT ) b = a ; 
        std::cout 
            << "morton_circle_demo"
            << " nbit " << std::setw(2) << nbit 
            << " mask (0x) " << std::setfill('0') << std::setw(16) << std::hex << mask << std::dec
            << " (0b) " << std::bitset<64>(mask).to_string()  
            << " a " << a->sstr() 
            << std::endl 
            ; 
    }

    if(b)
    {
        b->save(FOLD, "morton_circle_demo.npy"); 
        std::cout << " save " << FOLD << " " << b->sstr() << std::endl ; 
    }
}


int main(int argc, char** argv)
{
    morton_circle_demo();  
    return 0 ; 
}

/**
Range of nbit from 0->16 shows coarsening effect, beyond that it saturates and makes no difference. 

::

    epsilon:mortonlib blyth$ ./domain2d_test.sh
    morton_circle_demo nbit  0 mask (0x) 0000000000000000 (0b) 0000000000000000000000000000000000000000000000000000000000000000 a (1, 2, )
    morton_circle_demo nbit 01 mask (0x) 8000000000000000 (0b) 1000000000000000000000000000000000000000000000000000000000000000 a (2, 2, )
    morton_circle_demo nbit 02 mask (0x) c000000000000000 (0b) 1100000000000000000000000000000000000000000000000000000000000000 a (4, 2, )
    morton_circle_demo nbit 03 mask (0x) e000000000000000 (0b) 1110000000000000000000000000000000000000000000000000000000000000 a (8, 2, )
    morton_circle_demo nbit 04 mask (0x) f000000000000000 (0b) 1111000000000000000000000000000000000000000000000000000000000000 a (12, 2, )
    morton_circle_demo nbit 05 mask (0x) f800000000000000 (0b) 1111100000000000000000000000000000000000000000000000000000000000 a (16, 2, )
    morton_circle_demo nbit 06 mask (0x) fc00000000000000 (0b) 1111110000000000000000000000000000000000000000000000000000000000 a (20, 2, )
    morton_circle_demo nbit 07 mask (0x) fe00000000000000 (0b) 1111111000000000000000000000000000000000000000000000000000000000 a (28, 2, )
    morton_circle_demo nbit 08 mask (0x) ff00000000000000 (0b) 1111111100000000000000000000000000000000000000000000000000000000 a (36, 2, )
    morton_circle_demo nbit 09 mask (0x) ff80000000000000 (0b) 1111111110000000000000000000000000000000000000000000000000000000 a (56, 2, )
    morton_circle_demo nbit 10 mask (0x) ffc0000000000000 (0b) 1111111111000000000000000000000000000000000000000000000000000000 a (76, 2, )
    morton_circle_demo nbit 11 mask (0x) ffe0000000000000 (0b) 1111111111100000000000000000000000000000000000000000000000000000 a (116, 2, )
    morton_circle_demo nbit 12 mask (0x) fff0000000000000 (0b) 1111111111110000000000000000000000000000000000000000000000000000 a (156, 2, )
    morton_circle_demo nbit 13 mask (0x) fff8000000000000 (0b) 1111111111111000000000000000000000000000000000000000000000000000 a (248, 2, )
    morton_circle_demo nbit 14 mask (0x) fffc000000000000 (0b) 1111111111111100000000000000000000000000000000000000000000000000 a (368, 2, )
    morton_circle_demo nbit 15 mask (0x) fffe000000000000 (0b) 1111111111111110000000000000000000000000000000000000000000000000 a (516, 2, )
    morton_circle_demo nbit 16 mask (0x) ffff000000000000 (0b) 1111111111111111000000000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 17 mask (0x) ffff800000000000 (0b) 1111111111111111100000000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 18 mask (0x) ffffc00000000000 (0b) 1111111111111111110000000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 19 mask (0x) ffffe00000000000 (0b) 1111111111111111111000000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 20 mask (0x) fffff00000000000 (0b) 1111111111111111111100000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 21 mask (0x) fffff80000000000 (0b) 1111111111111111111110000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 22 mask (0x) fffffc0000000000 (0b) 1111111111111111111111000000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 23 mask (0x) fffffe0000000000 (0b) 1111111111111111111111100000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 24 mask (0x) ffffff0000000000 (0b) 1111111111111111111111110000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 25 mask (0x) ffffff8000000000 (0b) 1111111111111111111111111000000000000000000000000000000000000000 a (708, 2, )
    morton_circle_demo nbit 26 mask (0x) ffffffc000000000 (0b) 1111111111111111111111111100000000000000000000000000000000000000 a (708, 2, )


**/


