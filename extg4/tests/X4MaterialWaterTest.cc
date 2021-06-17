#include "OPTICKS_LOG.hh"
#include "X4MaterialWaterStandalone.hh"
#include "X4MaterialWater.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    X4MaterialWaterStandalone mws ; 
    //mws.dump(); 

    X4MaterialWater mw ; 
    //mw.dump(); 
    mw.rayleigh_scan(); 
    mw.rayleigh_scan2(); 

    // change the RAYLEIGH of "Water" to test that the below uses it without recalculating 
    mw.changeRayleighToMidBin();  

    // 2nd instanciation should be using the 
    // RAYLEIGH property added to the Water G4Material
    X4MaterialWater mw2 ; 
    mw2.rayleigh_scan2(); 

    return 0 ; 
}

//  X4MaterialWater=INFO X4MaterialWaterTest 


