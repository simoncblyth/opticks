// name=OpticksPhotonSTANDALONETest ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"

int main(int argc, char** argv)
{
    for(unsigned i=0 ; i < 16 ; i++) 
    {
        unsigned flag = 0x1 << i ; 
        std::cout 
            << " i " << std::setw(3) << i 
            << " flag " << std::setw(10) << flag 
            << " name " << std::setw(20) << OpticksPhoton::Flag(flag) 
            << std::endl 
            ;
    }
    return 0 ; 
}
