
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"


#ifdef WITH_OLD_FRAME
#include "sframe.h"
#else
#include "sfr.h"
#include "stree.h"
#endif

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

    CSGFoundry* fd = CSGFoundry::Load();
    const stree* tree = fd->getTree();

#ifdef WITH_OLD_FRAME
    sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m are set by CSGTarget::getFrame
    NP* gs0 = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr) ;
#else
    sfr fr = tree->get_frame_moi();
    NP* gs0 = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr) ;
#endif

    SEvt::AddGenstep(gs0);

    NP* gs = SEvt::GatherGenstep(SEvt::ECPU);

#ifdef WITH_OLD_FRAME
    NP* pp = SFrameGenstep::GenerateCenterExtentGenstepPhotons_( gs, fr.gridscale() );
#else
    NP* pp = SFrameGenstep::GenerateCenterExtentGenstepPhotons_( gs, fr.get_gridscale() );
    assert(  fr.get_gridscale()  > 0.f );
    //assert(0 && "NEED TO SET GRIDSCALE SOMEWHERE"); //
#endif



    std::cout << " fr " << std::endl << fr << std::endl ;

    gs->save("$FOLD/genstep.npy");
    pp->save("$FOLD/photon.npy");
    fr.save("$FOLD") ;

    return 0 ;
}

