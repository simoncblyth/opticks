// TEST=SSeqTest om-t


#include <iostream>
#include <iomanip>

#include "SSeq.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned long long s = 0x000d000c000b000a  ;

    SSeq<unsigned long long> seq(s) ;

    assert( seq.msn() == 0xd );

    for(unsigned i=0 ; i < 16 ; i++) 
       std::cout << std::setw(2) << i << " : " << std::hex << seq.nibble(i) << std::dec << std::endl ; 
 

    return 0 ;
}

