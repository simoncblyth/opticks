#ifdef OLD_PARAMETERS
#include "X_BParameters.hh"
#else
#include "NMeta.hpp"
#endif

#include "NOpenMeshCfg.hpp"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

#ifdef OLD_PARAMETERS
    X_BParameters meta ; 
#else
    NMeta meta ; 
#endif

    meta.add<std::string>("poly", "BSP");
    meta.add<std::string>("polycfg", "contiguous=1,reversed=0,numsubdiv=3,offsave=1");

    const char* treedir = NULL ; 

    NOpenMeshCfg cfg(&meta, treedir)  ;
    LOG(info) << cfg.desc() ;  

    return 0 ; 
}
