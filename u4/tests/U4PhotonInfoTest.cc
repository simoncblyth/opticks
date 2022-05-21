#include "U4PhotonInfo.h"
#include "U4Track.h"
#include <iostream>

void test_SetGet()
{
    G4Track* track = U4Track::MakePhoton(); 
    spho p0 = {1, 2, 3, 4} ; 
    U4PhotonInfo::Set(track, p0 ); 
    const G4Track* ctrack = track ; 
    spho p1 = U4PhotonInfo::Get(ctrack); 
    assert( p1.isIdentical(p0) ); 

    std::cout 
        << " track " << track << std::endl 
        << " p0 " << p0.desc() << std::endl 
        << " p1 " << p1.desc() << std::endl 
        ; 
}


int main()
{
    test_SetGet(); 


    return 0 ; 
}
