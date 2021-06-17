#include "OPTICKS_LOG.hh"
#include "X4MaterialWaterStandalone.hh"
#include "X4MaterialWater.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    X4MaterialWaterStandalone mws ; 
    mws.dump(); 

    X4MaterialWater mw ; 
    mw.dump(); 
    mw.rayleigh_scan(); 

    return 0 ; 
}


