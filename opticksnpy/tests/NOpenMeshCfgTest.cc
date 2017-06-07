#include "NOpenMeshCfg.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NOpenMeshCfg cfg ; 
    LOG(info) << cfg.desc() ;  

    return 0 ; 
}
