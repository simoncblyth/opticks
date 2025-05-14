#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "SEvt.hh"
#include "SGenerate.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEvt* evt = SEvt::Create(SEvt::ECPU) ;
    SEvt::AddTorchGenstep();

    NP* gs = evt->gatherGenstep();
    NP* ph = SGenerate::GeneratePhotons(gs);

    const char* _SGenerate__test_GS_NAME="SGenerate__test_GS_NAME" ;
    const char* _SGenerate__test_PH_NAME="SGenerate__test_PH_NAME" ;
    const char* GS_NAME = ssys::getenvvar(_SGenerate__test_GS_NAME, "gs.npy") ;
    const char* PH_NAME = ssys::getenvvar(_SGenerate__test_PH_NAME, "ph.npy") ;

    gs->save("$FOLD", GS_NAME);
    ph->save("$FOLD", PH_NAME);

    return 0 ;
}
