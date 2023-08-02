#include "NP.hh"
#include "smeta.h"

int main()
{
    NP* a = NP::Make<float>(1) ; 
    smeta::Collect(a->meta, "SEvt_AddEnvMeta_Test.cc/main" ); 

    a->save("$TMP/SEvt_AddEnvMeta_Test/a.npy") ;  
    
    return 0 ; 
}
