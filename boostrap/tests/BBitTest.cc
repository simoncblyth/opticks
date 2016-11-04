#include "BBit.hh"
#include <cassert>
#include <iostream>
#include <iomanip>

int main()
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


    return 0 ; 
}
