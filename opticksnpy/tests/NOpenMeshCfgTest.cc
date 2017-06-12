#include "NParameters.hpp"
#include "NOpenMeshCfg.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    NParameters meta ; 
    meta.add<std::string>("poly", "BSP");
    meta.add<std::string>("polycfg", "contiguous=1,reversed=0,numsubdiv=3,offsave=1");

    const char* treedir = NULL ; 

    NOpenMeshCfg cfg(&meta, treedir)  ;
    LOG(info) << cfg.desc() ;  

    return 0 ; 
}
