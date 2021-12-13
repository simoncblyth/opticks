#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"
#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    ok.configure(); 

    GGeo* ggeo = GGeo::Load(&ok); 

    const char* meta = nullptr ; 

    CSGFoundry foundry ; 
    CSG_GGeo_Convert conv(&foundry, ggeo, meta) ; 
    conv.convert(); 

    bool ops = SSys::getenvbool("ONE_PRIM_SOLID"); 
    if(ops) conv.addOnePrimSolid(); 

    bool ons = SSys::getenvbool("ONE_NODE_SOLID"); 
    if(ons) conv.addOneNodeSolid(); 

    bool dcs = SSys::getenvbool("DEEP_COPY_SOLID"); 
    if(dcs) conv.addDeepCopySolid(); 

    bool ksb = SSys::getenvbool("KLUDGE_SCALE_PRIM_BBOX"); 
    if(ksb) conv.kludgeScalePrimBBox();  

    const char* cfbase = SSys::getenvvar("CFBASE", "/tmp" ); 
    const char* rel = "CSGFoundry" ; 

    foundry.write(cfbase, rel );    // expects existing directory $CFBASE/CSGFoundry 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, rel);  // load foundary and check identical bytes
    assert( 0 == CSGFoundry::Compare(&foundry, fd ) );  

    return 0 ; 
}
