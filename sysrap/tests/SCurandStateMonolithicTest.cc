
#include <string>
#include <vector>
#include "OPTICKS_LOG.hh"
#include "srng.h"
#include "SCurandStateMonolithic.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info); 
#if defined(RNG_XORWOW)
    
    std::vector<std::string> specs = { "1:0:0", "3:0:0", "10:0:0" } ; 
    for(unsigned i=0 ; i < specs.size() ; i++)
    {
        const char* spec = specs[i].c_str(); 
        SCurandStateMonolithic scs(spec) ; 
        std::cout << scs.desc() << std::endl  ; 
    }

    LOG(info) << std::endl << SCurandStateMonolithic::Desc() ; 

#elif defined(RNG_PHILOX) || defined(RNG_PHILITEOX)
    LOG(info) 
        << argv[0] 
        << " NOT RELEVANT FOR srng<RNG>::NAME [" << srng<RNG>::NAME << "]" 
        << " CBPRNG : counter-based-pseudo-RNG " 
        ;  
#endif

    return 0 ; 
}
