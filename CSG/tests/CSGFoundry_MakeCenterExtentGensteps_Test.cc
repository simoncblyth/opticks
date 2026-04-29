
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sfr.h"
#include "stree.h"
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

    SEvt* sev = SEvt::Create_ECPU() ;

    CSGFoundry* fd = CSGFoundry::Load();
    const stree* tree = fd->getTree();
    sev->setTree(tree);   // THIS IS REQUIRED BEFORE SEvt::AddGenstep

    sfr fr = tree->get_frame_moi();
    NP* gs0 = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr) ;

    SEvt::AddGenstep(gs0);

    NP* gs = SEvt::GatherGenstep(SEvt::ECPU);

    NP* pp = SFrameGenstep::GenerateCenterExtentGenstepPhotons_( gs, fr.get_gridscale() );
    assert(  fr.get_gridscale()  > 0.f );
    //assert(0 && "NEED TO SET GRIDSCALE SOMEWHERE"); //

    std::cout << " fr " << std::endl << fr << std::endl ;

    gs->save("$FOLD/genstep.npy");
    pp->save("$FOLD/photon.npy");
    fr.save("$FOLD") ;

    return 0 ;
}

