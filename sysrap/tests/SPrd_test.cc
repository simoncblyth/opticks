
#include "NP.hh"
#include "SPrd.h"

int main(int argc, char** argv)
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard" ;
    const NP* bnd = NP::Load(base,"bnd.npy"); 
    SPrd* sprd = new SPrd(bnd) ;  
    std::cout << sprd->desc() ; 

    unsigned num_photon = 8 ; 
    unsigned num_bounce = 6 ; 
    NP* prd = sprd->duplicate_prd(num_photon, num_bounce); 

    prd->dump(); 
    prd->save("$FOLD/prd.npy"); 

    return 0 ; 
}
