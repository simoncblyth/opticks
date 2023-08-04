#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "spath.h"
#include "SGenerate.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(SEvt::ECPU) ; 
    SEvt::AddTorchGenstep(); 

    NP* gs = evt->gatherGenstep();     
    NP* ph = SGenerate::GeneratePhotons(gs); 

    const char* dir = spath::Resolve("$TMP/SGenerate_test"); 
    gs->save(dir, "gs.npy"); 
    ph->save(dir, "ph.npy"); 

    return 0 ; 
}
