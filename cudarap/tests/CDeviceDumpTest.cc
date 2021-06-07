#include <vector>
#include "CDevice.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    std::vector<CDevice> devs ; 

    const char* dirpath = nullptr ; 
    bool nosave = true ; 
    CDevice::Visible(devs, dirpath, nosave ); 
    CDevice::Dump( devs , "visible devices"); 
 
    return 0 ; 
}


