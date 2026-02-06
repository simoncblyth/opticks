#include "OPTICKS_LOG.hh"

#ifdef WITH_OLD_FRAME
#include "sframe.h"
#else
#include "sfr.h"
#endif

#include "SFrameGenstep.hh"
#include "NP.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    float extent = 100.f ;
#ifdef WITH_OLD_FRAME
    sframe fr ;
    fr.ce.w = extent ;
    NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr);
#else
    sfr fr = sfr::MakeFromExtent<float>(extent) ;
    NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr);
#endif

    LOG(info) << " gs " << ( gs ? gs->sstr() : "-" ) ;
    gs->save("$FOLD/gs.npy");

    return 0 ;

}
