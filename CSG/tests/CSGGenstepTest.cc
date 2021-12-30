#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
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
    bool ce_offset = SSys::getenvint("CE_OFFSET", 0) > 0 ; 
    gsm->create(moi, ce_offset);  
    gsm->generate_photons_cpu(); 
    gsm->save("$TMP/CSG/CSGGenstepTest"); 

    return 0 ; 
}

