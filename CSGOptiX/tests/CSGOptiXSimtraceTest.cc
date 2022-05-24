/**
CSGOptiXSimtraceTest : used from cxs.sh
===========================================

Using as much as possible the CSGOptiX rendering machinery 
to do simulation. Using CSGOptiX raygen mode 1 which flips the case statement 
in the __raygen__rg program. 

The idea is to minimize code divergence between ray trace rendering and 
ray trace enabled simulation. Because after all the point of the rendering 
is to be able to see the exact same geometry that the simulation is using.

::

     MOI=Hama CEGS=5:0:5:1000   CSGOptiXSimtraceTest
     MOI=Hama CEGS=10:0:10:1000 CSGOptiXSimtraceTest

TODO: 
    find way to get sdf distances for all intersects GPU side 
    using GPU side buffer of positions 

    just need to run the distance function for all points and with 
    the appropriate CSGNode root and num_node

    does not need OptiX, just CUDA (or it can be done CPU side also)

**/

#include <cuda_runtime.h>

#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"

#include "NP.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SOpticks.hh"
#include "SEvt.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "QEvent.hh"

#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"
#include "QSim.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SEventConfig::SetRGMode("simtrace"); 
    SOpticks::WriteOutputDirScript() ; // writes CSGOptiXSimtraceTest_OUTPUT_DIR.sh in PWD 
   
    SEvt evt ; 
 
    CSGFoundry* fd = CSGFoundry::Load(); 
    if(fd->hasMeta()) LOG(info) << "fd.meta\n" << fd->meta ; 
    LOG(info) << "foundry " << fd->desc() ; 

    CSGOptiX* cx = CSGOptiX::Create(fd); 
    QSim* qs = cx->sim ; 

    // TODO: rejig the below to be more like torch or carrier gensteps 
    {
        CSGGenstep* gsm = fd->genstep ;    // THIS IS THE GENSTEP MAKER : NOT THE GS THEMSELVES 
        const char* moi = SSys::getenvvar("MOI", "sWorld:0:0");  
        bool ce_offset = SSys::getenvint("CE_OFFSET", 0) > 0 ; 
        bool ce_scale = SSys::getenvint("CE_SCALE", 0) > 0 ;   
        gsm->create(moi, ce_offset, ce_scale ); // SEvent::MakeCenterExtentGensteps
        NP* gs = gsm->gs ; 
        sframe fr ; 
        fr.ce = gsm->ce ; 
        qat4::copy(fr.m2w, *gsm->m2w); 
        qat4::copy(fr.w2m, *gsm->w2m); 
        cx->setFrame(fr); 
        cx->setCEGS(gsm->cegs);   // sets peta metadata
        cx->setMetaTran(gsm->geotran); 
        SEvt::AddGenstep(gs); 
    }

    qs->simtrace();  

    cudaDeviceSynchronize(); 

    //cx->snapSimtraceTest();  
    qs->save(); // uses SGeo::LastUploadCFBase_OutDir to place outputs into CFBase/ExecutableName folder sibling to CSGFoundry   

    const char* dir = QEvent::DefaultDir(); 
    cx->fr.save(dir);  


 
    return 0 ; 
}
