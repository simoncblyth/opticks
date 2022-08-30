#include "OPTICKS_LOG.hh"

#include "sframe.h"
#include "SFrameGenstep.hh"
#include "NP.hh"

const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    sframe fr ;  
    fr.ce.w = 100.f ; 

    NP* gs = SFrameGenstep::MakeCenterExtentGensteps(fr);
    LOG(info) << " gs " << ( gs ? gs->sstr() : "-" ) ;  
    gs->save(FOLD, "gs.npy"); 

    return 0 ; 

}
