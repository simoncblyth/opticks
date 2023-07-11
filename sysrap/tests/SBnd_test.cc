
#include "NPFold.h"
#include "SBnd.h"
#include "ssys.h"


int main(int argc, char** argv)
{
    const char* base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ;

    //const char* relp = "extra/GGeo" ;    
    const char* relp = "stree/standard" ; 
    const NP* bnd     = NP::Load(base, relp, "bnd.npy"); 
    const NP* optical = NP::Load(base, relp, "optical.npy"); 

    if( bnd == nullptr ) 
    { 
        std::cerr 
            << " FAILED to load bnd.npy" 
            << " base " << base 
            << " relp " << relp 
            << " : PROBABLY GEOM envvar is not defined " 
            << std::endl 
            ; 
        return 1 ; 
    }

    std::cout << " bnd " << ( bnd ? bnd->sstr() : "-" ) << std::endl ; 
    std::cout << " optical " << ( optical ? optical->sstr() : "-" ) << std::endl ; 


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


    const char* bnd_fallback = R"LITERAL(
    Acrylic///LS
    Water///Acrylic
    Water///Pyrex
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    )LITERAL" ; 

    // HMM: why did the bnd change ? Boundary to skin ?  
    // Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
        
    const char* bnd_sequence = ssys::getenvvar("BND_SEQUENCE", bnd_fallback );
    std::cout << " bnd_sequence " << bnd_sequence << std::endl ;

    std::vector<unsigned> bnd_idx ;
    sb.getBoundaryIndices( bnd_idx, bnd_sequence, '\n' );
    std::cout << "sb.descBoundaryIndices" << std::endl << sb.descBoundaryIndices( bnd_idx ) << std::endl ;


    return 0 ; 
}
