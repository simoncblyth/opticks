#include "SSys.hh"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGGenstep.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    LOG(info) << "cfbase " << cfbase ; 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    CSGGenstep* gsm = fd->genstep ; 
    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0");  
    gsm->create(moi);  


    return 0 ; 
}


