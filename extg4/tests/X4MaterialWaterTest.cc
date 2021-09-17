#include "OPTICKS_LOG.hh"
#include "X4MaterialWaterStandalone.hh"
#include "X4MaterialWater.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << " create Water G4Material with X4MaterialWaterStandalone " ; 
    X4MaterialWaterStandalone mws ; 
    //mws.dump(); 

    X4MaterialWater mw ; 
    //mw.dump(); 

    LOG(info) << " mw.X4MaterialWater::rayleigh_scan " ; 
    mw.rayleigh_scan(); 

    LOG(info) << " mw.X4MaterialWater::rayleigh_scan2 " ; 
    mw.rayleigh_scan2(); 

    LOG(info) << " mw.X4MaterialWater::changeRayleighToMidBin : messing up the values as a check they are not recalculated  " ; 
    mw.changeRayleighToMidBin();  


    LOG(info) << " mw2.X4MaterialWater : 2nd instanciation should be using the messed up property without recalculation " ; 
    X4MaterialWater mw2 ; 

    LOG(info) << " mw2.X4MaterialWater::rayleigh_scan2 " ; 
    mw2.rayleigh_scan2(); 

    return 0 ; 
}

//  X4MaterialWater=INFO X4MaterialWaterTest 


