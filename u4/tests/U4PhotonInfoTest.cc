#include "U4PhotonInfo.h"
#include "U4Track.h"
#include <iostream>


int main()
{
    G4Track* track = U4Track::MakePhoton(); 
    std::cout << "track " << track << std::endl ; 

    //p->SetUserInformation(U4PhotonInfo::MakeScintillation(gs, i, ancestor ));

    spho pho = {1, 2, 3, 4} ; 

    U4PhotonInfo* pin = new U4PhotonInfo(pho) ; 
    track->SetUserInformation(pin);

    std::cout << " pin " << pin->desc() << std::endl ;  


    return 0 ; 
}
