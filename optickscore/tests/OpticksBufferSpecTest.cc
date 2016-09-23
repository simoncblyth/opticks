#include "OpticksBufferSpec.hh"
#include "OpticksEvent.hh"
#include "OpticksCMakeConfig.hh"
#include "OpticksSwitches.h"

#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

#ifdef WITH_SEED_BUFFER
    LOG(info) << "WITH_SEED_BUFFER" ;
#else
    LOG(info) << "NOT(WITH_SEED_BUFFER)" ;
#endif

    LOG(info) << "OXRAP_OPTIX_VERSION : " << OXRAP_OPTIX_VERSION ;
    LOG(info) << "CFG4_G4VERSION_NUMBER : " << CFG4_G4VERSION_NUMBER ;

    std::vector<std::string> names ; 
    OpticksEvent::pushNames(names);

    for(unsigned j=0 ; j < 2 ; j++)
    {
        bool compute = j == 0 ; 
        std::cout << std::endl <<  ( compute ? "COMPUTE" : "INTEROP" ) << std::endl  ; 
        for(unsigned i=0 ; i < names.size() ; i++)
            std::cout 
                      << std::setw(20) <<  names[i] << " : " 
                      << OpticksBufferSpec::Get(names[i].c_str(), compute  )
                      << std::endl 
                      ; 
                     
    }
    return 0 ; 
}
