

#include  <iostream>
#include "PLOG.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"

#include "Randomize.hh"

#include "CG4.hh"
#include "CFG4_LOG.hh"
#include "CRandomEngine.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG__ ; 

    LOG(info) << argv[0] ; 


    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );   // with GPU running this is COMPUTE/INTEROP

    OpticksHub hub(&ok) ; 
    
    CG4* g4 = new CG4(&hub) ; 


    CRandomEngine rng(g4) ; 

    for(int i=0 ; i < 10 ; i++)
    {
        //std::cout << rng.flat() << std::endl  ;  
        std::cout << G4UniformRand() << std::endl  ;  
    }    
 

    return 0 ; 
}


