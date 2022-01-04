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

    const char* argforced = "--gparts_transform_offset" ; 
    Opticks ok(argc, argv, argforced );
    ok.configure(); 

    const char* idpath = ok.getIdPath() ; 

    LOG(error) << "[ load ggeo from idpath " << idpath  ; 
    GGeo* ggeo = GGeo::Load(&ok); 
    LOG(error) << "] load ggeo " ; 


    CSGFoundry foundry ; 

    LOG(error) << "[ convert ggeo " ; 
    CSG_GGeo_Convert conv(&foundry, ggeo ) ; 
    conv.convert(); 
    LOG(error) << "] convert ggeo " ; 


    bool ops = SSys::getenvbool("ONE_PRIM_SOLID"); 
    if(ops) conv.addOnePrimSolid(); 

    bool ons = SSys::getenvbool("ONE_NODE_SOLID"); 
    if(ons) conv.addOneNodeSolid(); 

    bool dcs = SSys::getenvbool("DEEP_COPY_SOLID"); 
    if(dcs) conv.addDeepCopySolid(); 

    bool ksb = SSys::getenvbool("KLUDGE_SCALE_PRIM_BBOX"); 
    if(ksb) conv.kludgeScalePrimBBox();  


    const char* cfbase = ok.getFoundryBase("CFBASE"); 
    const char* rel = "CSGFoundry" ; 

    LOG(error) << "[ write foundry to CFBASE " << cfbase << " rel " << rel  ; 
    foundry.write(cfbase, rel );   
    LOG(error) << "] write foundry " ; 

    LOG(error) << "[ load foundry " ; 
    CSGFoundry* fd = CSGFoundry::Load(cfbase, rel);  // load foundary and check identical bytes
    LOG(error) << "] load foundry " ; 

    assert( 0 == CSGFoundry::Compare(&foundry, fd ) );  

    LOG(info) << "CSGFoundry saved to cfbase " << cfbase ; 
    LOG(info) << "logs are written to logdir " << logdir ; 
    return 0 ; 
}
