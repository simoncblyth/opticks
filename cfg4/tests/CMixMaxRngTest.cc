#include "OPTICKS_LOG.hh"
#include "Randomize.hh"
#include "CMixMaxRng.hh"


void dump_flat(int n)
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;

    long seed = engine->getSeed() ; 
    LOG(info) 
        << " seed " << seed 
        << " name " << engine->name() 
        ; 

    for(int i=0 ; i < n ; i++)
    {
        double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
        std::cout << u << std::endl ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    CMixMaxRng mmr ;
    dump_flat(10); 

    return 0 ; 
}


