
#include "NP.hh"
#include "SBnd.h"

int main(int argc, char** argv)
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ;

    NP* bnd = NP::Load(base, "bnd.npy") ; 

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

    return 0 ; 
}
