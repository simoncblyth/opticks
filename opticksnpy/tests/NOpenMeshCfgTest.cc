#include "NParameters.hpp"
#include "NOpenMeshCfg.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NParameters meta ; 
    meta.add<std::string>("poly", "BSP");
    meta.add<std::string>("polycfg", "contiguous=1,reversed=0,numsubdiv=3,offsave=1");

    const char* treedir = NULL ; 

    NOpenMeshCfg cfg(&meta, treedir)  ;
    LOG(info) << cfg.desc() ;  

    return 0 ; 
}
