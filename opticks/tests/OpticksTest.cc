// ggv --opticks 

#include "Opticks.hh"
#include <iostream>

int main(int argc, char** argv)
{
    Opticks ok(argc, argv);
    ok.Summary();

    unsigned long long seqmat = 0x0123456789abcdef ;
    std::cout << " seqmat "
              << std::hex << seqmat << std::dec
              << Opticks::MaterialSequence(seqmat) 
              << std::endl 
              ;

    return 0 ;
}
