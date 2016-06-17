#include "BBit.hh"
#include <iostream>
#include <iomanip>

int main()
{
    for(int i=0 ; i < 64 ; i++ )
    {
        int msk = 1 << i ; 
        int ffs_ = BBit::ffs(msk);   
        std::cout << " msk " << std::hex << std::setw(10) << msk  
                  << " ffs " << std::dec << std::setw(10) << ffs_
                  << std::endl ;   

    } 


    return 0 ; 
}
