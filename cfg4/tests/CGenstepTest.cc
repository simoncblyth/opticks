
//#include <boost/dynamic_bitset.hpp>

#include "OPTICKS_LOG.hh"
#include "CGenstep.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
 
    unsigned index = 0 ; 
    unsigned photons = 100 ; 
    unsigned offset = 1000 ; 
    char gentype = 'C' ; 

    CGenstep cg(index, photons, offset, gentype); 
    for(int i=0 ; i < int(photons) ; i++) if(i % 10 == 0 ) cg.set(i) ; 

    LOG(info) << cg.desc() ;

    return 0 ; 
}

// om- ; TEST=CGenstepTest om-t
