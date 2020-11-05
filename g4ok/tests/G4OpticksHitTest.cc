#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "NPY.hpp"
#include "GGeo.hh"
#include "GPho.hh"
//#include "G4OpticksHit.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    Opticks ok(argc, argv);

    const char* default_path = "$TMP/G4OKTest/evt/g4live/natural/1/ox.npy" ; 
    const char* path = argc > 1 ? argv[1] : default_path ;   

    NPY<float>* ox = NPY<float>::load( path ) ; 
    if(ox == NULL ) return 0 ; 

    // load full geometry in order to have access to the transforms for local positions
    // hmm: could consider just loading GNodeLib ?
    GGeo* gg = GGeo::Load(&ok); 
    GPho ph(gg);   
    ph.setPhotons(ox);
    ph.setSelection('H');  // A:All L:Landed H:Hit  
    
    unsigned maxDump = 0 ; 
    ph.dump("G4OpticksHitTest", maxDump); 


    return 0 ; 
}





