
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"
#include "spath.h"

#include "SSim.hh"
#include "SEvt.hh"
#include "SFrameGenstep.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt::Create_ECPU() ;

    SSim::Create();   // this is creating (scontext)sctx : checking GPU 



    CSGFoundry* fd = CSGFoundry::Load();

    sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m are set by CSGTarget::getFrame 

    NP* gs0 = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr) ; 
    SEvt::AddGenstep(gs0); 

    NP* gs = SEvt::GatherGenstep(SEvt::ECPU); 
    NP* pp = SFrameGenstep::GenerateCenterExtentGenstepPhotons_( gs, fr.gridscale() );  

    std::cout << " fr " << std::endl << fr << std::endl ; 

    gs->save("$FOLD/genstep.npy"); 
    pp->save("$FOLD/photon.npy"); 
    fr.save("$FOLD") ; 

    return 0 ; 
}

