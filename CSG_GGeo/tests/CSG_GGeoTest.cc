#include "SSys.hh"
#include "SPath.hh"
#include "BOpticksResource.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"
#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"

int main(int argc, char** argv)
{
    const char* logdir = BOpticksResource::GetCachePath("CSG_GGeo/logs"); 
    std::cout << "change directory to logdir " << logdir << std::endl ; 
    SPath::chdir(logdir); 

    OPTICKS_LOG(argc, argv);

    const char* argforced = "--gparts_transform_offset --savegparts " ; 
    Opticks ok(argc, argv, argforced );
    ok.configure(); 

    bool gparts_transform_offset = ok.isGPartsTransformOffset() ; 
    if(gparts_transform_offset == false)
    {
        LOG(fatal) 
            << " GParts geometry requires use of --gparts_transform_offset "
            << " for interoperation with the CSGFoundry single array of transforms approach "
            << " failing to use this results in incorrect transforms "
            ;
    }
    assert(gparts_transform_offset == true ); 


    unsigned numCXSkipLV = ok.getNumCXSkipLV();  // --cxskiplv 1,101,202
    LOG(info) << " numCXSkipLV " << numCXSkipLV ; 
      
    const char* idpath = ok.getIdPath() ; 

    LOG(error) << "[ load ggeo from idpath " << idpath  ; 
    GGeo* ggeo = GGeo::Load(&ok); 
    LOG(error) << "] load ggeo " ; 

    if(ok.isEarlyExit())
    {
        LOG(fatal) << " early exit due to --earlyexit option " << argv[0] ; 
        return 0 ; 
    }  

    CSGFoundry* fd0 = CSG_GGeo_Convert::Translate(ggeo); 
    assert( fd0->sim ); 

    const char* cfbase = ok.getFoundryBase("CFBASE"); 

    LOG(error) << "[ write foundry to CFBASE " << cfbase   ; 
    fd0->write(cfbase, "CSGFoundry" );   
    LOG(error) << "] write foundry " ; 

    LOG(error) << "[ load foundry " ; 
    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry");  // load foundary and check identical bytes
    assert( fd->sim ); 
    LOG(error) << "] load foundry " ; 

    assert( 0 == CSGFoundry::Compare(fd0, fd ) );  

    LOG(info) << "CSGFoundry saved to cfbase " << cfbase ; 
    LOG(info) << "logs are written to logdir " << logdir ; 
    return 0 ; 
}
