#include "OPTICKS_LOG.hh"

#include "sfr.h"
#include "SFrameGenstep.hh"
#include "NP.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    float extent = 100.f ;
    sfr fr = sfr::MakeFromExtent<float>(extent) ;
    NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(fr);

    LOG(info) << " gs " << ( gs ? gs->sstr() : "-" ) ;
    gs->save("$FOLD/gs.npy");

    return 0 ;

}
