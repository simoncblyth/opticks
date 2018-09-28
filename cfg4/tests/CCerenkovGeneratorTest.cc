// TEST=CCerenkovGeneratorTest om-t

#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "NPY.hpp"
#include "GGeo.hh"
#include "GBndLib.hh"

#include "OpticksGenstep.hh"
#include "CCerenkovGenerator.hh"
#include "C4PhotonCollector.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"
#include "CMaterialLib.hh"
#include "CAlignEngine.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );  

    OpticksHub hub(&ok) ; 
    CMaterialLib* clib = new CMaterialLib(&hub);
    clib->convert();
    // TODO: a more direct way to get to a refractive index, than the above that loads the entire geometry  


    // below needs to be done after Opticks::configure for setup of the event spec

    NPY<float>* np = ok.hasKey() ? ok.loadDirectGenstep() : NULL ;   
    if(np == NULL) return 0 ; 

    OpticksGenstep* gs = new OpticksGenstep(np) ; 
    unsigned modulo = 1000 ; 
    unsigned margin = 10 ;   
    gs->dump( modulo, margin ) ; 

    const char* testdir = "$KEYDIR/tests/CCerenkovGeneratorTest" ;
    CAlignEngine::Initialize( testdir );  

    unsigned idx = 0 ;  
    G4VParticleChange* pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(gs,idx) ;

    C4PhotonCollector* collector = new C4PhotonCollector ; 
    collector->collectSecondaryPhotons( pc, idx ); 


    NPYBase::SetNPDump(true);
    collector->savePhotons(testdir, "so.npy") ; 
    NPYBase::SetNPDump(false);

    LOG(info) << collector->desc() ;

    //SSys::npdump( ph_path, "np.float32", "", "suppress=True") ;  

    return 0 ; 
}

