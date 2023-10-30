/**
sdevice_test.cc
==================

See notes in sdevice_test.sh 

**/


#include "sdevice.h"

int main(int argc, char** argv)
{  
    std::vector<sdevice> devs ; 
    sdevice::Visible(devs, "$HOME/.opticks/runcache", false );  
    std::cout << sdevice::Desc( devs ) ; 
    //std::cout << sdevice::Brief( devs ) ; 

    return 0 ; 
}
