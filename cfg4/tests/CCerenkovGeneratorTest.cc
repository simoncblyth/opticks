// TEST=CCerenkovGeneratorTest om-t

#include "OPTICKS_LOG.hh"

#include "NPY.hpp"
#include "NGS.hpp"
#include "CCerenkovGenerator.hh"


#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"
#include "CMaterialLib.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ; 
    const char* path = argc > 1 ? argv[1] : def ; 

    NPY<float>* np = NPY<float>::load(path) ; 
    if(np == NULL) return 0 ; 

    NGS* gs = new NGS(np) ; 
    unsigned modulo = 1000 ; 
    unsigned margin = 10 ;   
    gs->dump( modulo, margin ) ; 


    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4 
    OpticksHub hub(&ok) ; 
    CMaterialLib* clib = new CMaterialLib(&hub);
    clib->convert();
    // TODO: a more direct way to get to a refractive index 

 
    CCerenkovGenerator* cg = new CCerenkovGenerator(gs) ; 
    G4VParticleChange* pc = cg->generatePhotonsFromGenstep(0);
    LOG(info) << " pc " << pc ; 


    return 0 ; 
}
