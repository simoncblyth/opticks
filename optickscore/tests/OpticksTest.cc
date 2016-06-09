// ggv --opticks 

#include "Opticks.hh"
#include <iostream>
#include "BLog.hh"


int main(int argc, char** argv)
{
    Opticks ok(argc, argv);
    ok.Summary();

    ok.configure();

    unsigned long long seqmat = 0x0123456789abcdef ;
    LOG(info) << "OpticksTest::main"
              << " seqmat "
              << std::hex << seqmat << std::dec
              << " MaterialSequence " 
              << Opticks::MaterialSequence(seqmat) 
              ;

    return 0 ;
}
