
#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NSnapConfig.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* snapconfig = "steps=5,prefix=/some/dir/base,postfix=.ppm" ; 

    NSnapConfig cfg(snapconfig);
    cfg.dump();

    assert( cfg.steps == 5 );
  
    for(int i=0 ; i < cfg.steps ; i++)
    {
        std::cout << cfg.getSnapPath(i) << std::endl ; 
    }


    return 0 ; 
}
