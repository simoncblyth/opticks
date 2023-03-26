#include "OPTICKS_LOG.hh"
#include "NP.hh"
#include "SEvt.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NP* a = NP::Make<float>(1) ; 
    SEvt::AddEnvMeta(a, true) ; 

    a->save("$TMP/SEvt_AddEnvMeta_Test/a.npy") ;  
    
    return 0 ; 
}
