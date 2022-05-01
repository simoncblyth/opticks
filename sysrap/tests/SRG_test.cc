// name=SRG_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <vector>
#include <iostream>
#include <iomanip>

#include "SRG.h"

int main()
{
    std::vector<unsigned> raygenmode = {SRG_RENDER, SRG_SIMTRACE, SRG_SIMULATE} ; 
    for(unsigned i=0 ; i < raygenmode.size() ; i++)
    {
        std::cout << std::setw(5) << raygenmode[i] << std::setw(20) << SRG::Name(raygenmode[i]) << std::endl ;
    }
    return 0 ; 
}
