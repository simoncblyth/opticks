
#include "spho.h"
#include "U4Track.h"
#include "U4TrackInfo.h"


int main(int argc, char** argv)
{
    G4Track* track = U4Track::MakePhoton(); 
    spho p0 = {1, 2, 3, {10,20,30,40}} ; 

    U4TrackInfo<spho>::Set(track, p0 ); 
    const G4Track* ctrack = track ; 

    std::cout << U4Track::Desc<spho>(ctrack) << std::endl ; 

    spho p1 = U4TrackInfo<spho>::Get(ctrack); 

    assert( p1.isIdentical(p0) ); 

    spho* p2 = U4TrackInfo<spho>::GetRef(ctrack); 

    assert( p2->isIdentical(p0) ); 

    p2->uc4.w = 'Z' ; 

    std::cout << U4Track::Desc<spho>(ctrack) << std::endl ; 


    return 0 ; 
}

