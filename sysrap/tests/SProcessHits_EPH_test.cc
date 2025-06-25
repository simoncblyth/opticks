#include "SProcessHits_EPH.h"

int main()
{
    SProcessHits_EPH* eph = new SProcessHits_EPH ;

    for(int i=0 ; i < 1000000 ; i++) eph->add( EPH::SAVENORM, i % 2 == 0 ? true : false );

    eph->EndOfEvent_Simulate_merged_count = 10000 ;
    eph->EndOfEvent_Simulate_savehit_count = 20000 ;

    std::cout << eph->desc() ;

    NP* meta = eph->get_meta_array();
    meta->save("$FOLD/eph.npy");

    return 0 ;
}
