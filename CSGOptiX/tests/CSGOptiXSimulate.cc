/**
CSGOptiXSimulate
=================

**/

#include <algorithm>
#include <iterator>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

#include "QCtx.hh"


int main(int argc, char** argv)
{
    for(int i=0 ; i < argc ; i++ ) std::cout << argv[i] << std::endl; 

    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 
    ok.setRaygenMode(1) ; // override --raygenmode option 

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    // create GPU textures 
    QCtx<float>::Init(fd->icdf, fd->bnd ); 


    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 
    cx.setTop(top); 

    if( cx.raygenmode == 0 )
    {
        LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::render cx.raygenmode " << cx.raygenmode ; 
        assert(0); 
    }

    double dt = cx.simulate();  
    LOG(info) << " dt " << dt ;
 
    // TODO : save gensteps and photons

    return 0 ; 
}
