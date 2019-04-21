/**

NSnapConfigTest --snapconfig "numsteps=10,eyestartz=-1,eyestopz=1" 
NSnapConfigTest --snapconfig "numsteps=5,eyestartz=-1,eyestopz=-1" 


**/

#include "OPTICKS_LOG.hh"
#include "NSnapConfig.hpp"

#include "S_get_option.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* fallback = "steps=5,prefix=/some/dir/base,postfix=.ppm" ; 
    std::string snapconfig = get_option<std::string>(argc, argv, "--snapconfig", fallback ) ;

    NSnapConfig cfg(snapconfig.c_str());
    cfg.dump();
  
    for(int i=0 ; i < cfg.steps ; i++)
    {
        std::cout << cfg.getSnapPath(i) << std::endl ; 
    }


    return 0 ; 
}
