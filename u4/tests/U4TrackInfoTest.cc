#include <csignal>
#include "spho.h"
#include "U4Track.h"
#include "STrackInfo.h"


int main(int argc, char** argv)
{
    G4Track* track = U4Track::MakePhoton();
    spho p0 = {1, 2, 3, {10,20,30,40}} ;

    STrackInfo::Set(track, p0 );
    const G4Track* ctrack = track ;

    std::cout << U4Track::Desc(ctrack) << std::endl ;

    spho p1 = STrackInfo::Get(ctrack);

    bool p1_expect = p1.isIdentical(p0) ;
    assert( p1_expect );
    if(!p1_expect) std::raise(SIGINT);

    spho* p2 = STrackInfo::GetRef(ctrack);

    bool p2_expect = p2->isIdentical(p0) ;
    assert( p2_expect );
    if(!p2_expect) std::raise(SIGINT);

    p2->uc4.w = 'Z' ;

    std::cout << U4Track::Desc(ctrack) << std::endl ;


    return 0 ;
}

