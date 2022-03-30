// ./OpticksPhotonSTANDALONETest.sh

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
            << " OpticksPhoton::Flag " << std::setw(20)  << OpticksPhoton::Flag(flag) 
            << " OpticksPhoton::Abbrev " << std::setw(4) << OpticksPhoton::Abbrev(flag) 
            << std::endl 
            ;
    }
    return 0 ; 
}
