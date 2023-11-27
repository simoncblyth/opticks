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

    NP* gs = SFrameGenstep::MakeCenterExtentGenstep(fr);
    LOG(info) << " gs " << ( gs ? gs->sstr() : "-" ) ;  

    if(FOLD)
    {
        gs->save(FOLD, "gs.npy"); 
    }
    else
    {
        LOG(error) << "define FOLD envvar to save the gensteps array " ; 
    }

    return 0 ; 

}
