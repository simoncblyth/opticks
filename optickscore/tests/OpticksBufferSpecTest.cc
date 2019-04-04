#include "OpticksBufferSpec.hh"
#include "OpticksEvent.hh"
//#include "OpticksCMakeConfig.hh"
#include "OKConf_Config.hh"
#include "OpticksSwitches.h"

#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

#ifdef OXRAP_OPTIX_VERSION    
    LOG(info) << "OXRAP_OPTIX_VERSION : " << OXRAP_OPTIX_VERSION ;
#endif

#ifdef CFG4_G4VERSION_NUMBER
    LOG(info) << "CFG4_G4VERSION_NUMBER : " << CFG4_G4VERSION_NUMBER ;
#endif

#ifdef OKCONF_OPTIX_VERSION_INTEGER
    LOG(info) << "OKCONF_OPTIX_VERSION_INTEGER  : " << OKCONF_OPTIX_VERSION_INTEGER ; 
#endif

#ifdef OKCONF_GEANT4_VERSION_INTEGER
    LOG(info) << "OKCONF_GEANT4_VERSION_INTEGER : " << OKCONF_GEANT4_VERSION_INTEGER ; 
#endif


#ifdef WITH_SEED_BUFFER
    LOG(info) << "WITH_SEED_BUFFER" ;
#else
    LOG(info) << "NOT(WITH_SEED_BUFFER)" ;
#endif


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
