#include "sdevice.h"

int main(int argc, char** argv)
{  
    std::vector<sdevice> devs ; 

    const char* dirpath = "$HOME/.opticks/runcache" ; 
    sdevice::Visible(devs, dirpath, nosave );  
    sdevice::Dump( devs , "visible devices"); 

    return 0 ; 
}
