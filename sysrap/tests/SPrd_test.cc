
#include "spath.h"
#include "NP.hh"
#include "SPrd.h"

int main(int argc, char** argv)
{
    SPrd* sprd = SPrd::Load() ;
    if(!sprd) std::cerr << "FAILED TO SPrd::Load\n" ;
    if(!sprd) return 1 ;

    std::cout << sprd->desc() ;

    unsigned num_photon = 8 ;
    unsigned num_bounce = 6 ;
    NP* prd = sprd->fake_prd(num_photon, num_bounce);

    prd->dump();
    prd->save("$FOLD/prd.npy");

    return 0 ;
}
