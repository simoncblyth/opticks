
#include <iostream>
#include <cassert>
#include "CFG4_BODY.hh"
#include "G4ThreeVector.hh"

#include "PLOG.hh"
#include "CFG4_LOG.hh"

int main(int, char** )
{
    //PLOG_(argc, argv);
    //CFG4_LOG_ ;

    G4ThreeVector o(1,2,3);

    std::cout << o << std::endl ; 

    return 0 ; // (*lldb*) Exit
}

