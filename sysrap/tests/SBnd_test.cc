
#include "NPFold.h"
#include "SBnd.h"

int main(int argc, char** argv)
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ;

    // NB these hail from old world GGeo/X4 geometry conversion 
    const NP* bnd = NP::Load(base, "bnd.npy"); 
    const NP* optical = NP::Load(base, "optical.npy"); 

    if( bnd == nullptr ) 
    { 
        std::cerr 
            << " FAILED to load bnd.npy from base " << base 
            << " : PROBABLY GEOM envvar is not defined " 
            << std::endl 
            ; 
        return 1 ; 
    }

    SBnd sb(bnd) ; 
    std::cout << sb.desc() ;  

    NP* bd = sb.bd_from_optical(optical) ; 
    NP* mat = sb.mat_from_bd(bd) ; 


    NPFold* fold = new NPFold ; 

    fold->add("bnd", bnd ) ; 
    fold->add("optical", optical ) ; 
    fold->add("bd", bd ) ; 
    fold->add("mat", mat ) ; 

    fold->save("$FOLD") ; 

    return 0 ; 
}
