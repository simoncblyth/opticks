#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SPath.hh"
#include "SGenerate.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 
    SEvt::AddTorchGenstep(); 

    NP* gs = SEvt::GetGenstep();     
    NP* ph = SGenerate::GeneratePhotons(gs); 

    const char* dir = SPath::Resolve("$TMP/SGenerate_test", DIRPATH); 
    gs->save(dir, "gs.npy"); 
    ph->save(dir, "ph.npy"); 

    return 0 ; 
}
