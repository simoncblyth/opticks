// TEST=BBitTest om-t

#include "BBit.hh"
#include <cassert>
#include <iostream>
#include <iomanip>

void test_ffs()
{
    for(int i=0 ; i < 64 ; i++ )
    {
        int msk = 0x1 << i ; 
        int chk = BBit::ffs(msk) - 1;   
        std::cout 
                  << " msk ( 0x1 << " << std::setw(2) << std::dec << i << ") = "  << std::hex << std::setw(10) << msk  
                  << " BBit::ffs(msk) - 1 =  " << std::dec << std::setw(10) << chk
                  << std::endl ;   

         if(i < 32 ) assert(  chk == i );

    } 
}

void test_count_nibbles()
{

    typedef unsigned long long ULL ; 

    ULL msk = 0ull ;  
    for(ULL i=0 ; i < 64ull ; i++ )
    {
        msk |= 0x1ull << i ;  
        ULL nn = BBit::count_nibbles(msk); 
        std::cout 
             << " msk 0x " << std::hex << std::setw(16) << msk  
             << " nibbles " << std::dec << std::setw(4) << nn
             << std::endl 
             ;
    }
}



int main()
{
    //test_ffs(); 
    test_count_nibbles(); 

    return 0 ; 
}
