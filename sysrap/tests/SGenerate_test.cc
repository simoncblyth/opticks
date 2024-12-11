/**

~/o/sysrap/tests/SGenerate_test.sh 

**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SGenerate.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* evt = SEvt::Create(SEvt::ECPU) ; 
    SEvt::AddTorchGenstep(); 

    NP* gs = evt->gatherGenstep();     
    NP* ph = SGenerate::GeneratePhotons(gs); 

    gs->save("$FOLD/gs.npy"); 
    ph->save("$FOLD/ph.npy"); 

    return 0 ; 
}
