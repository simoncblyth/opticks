#include "sdevice.h"
#include "spath.h"

int main(int argc, char** argv)
{  
    std::vector<sdevice> devs ; 

    bool nosave = false ; 
    const char* dirpath = spath::ResolvePath("$HOME/.opticks/runcache") ; 

    sdevice::Visible(devs, dirpath, nosave );  
    sdevice::Dump( devs , "visible devices"); 

    return 0 ; 
}
