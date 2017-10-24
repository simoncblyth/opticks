#include <iostream>
#include <iomanip>

#include "SSeq.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    unsigned long long s = 0x000d000c000b000a  ;

    SSeq<unsigned long long> seq(s) ;

    assert( seq.msn() == 0xd );

    for(unsigned i=0 ; i < 16 ; i++) 
       std::cout << std::setw(2) << i << " : " << std::hex << seq.nibble(i) << std::dec << std::endl ; 
 

    return 0 ;
}

