
#include "ssys.h"
#include "spho.h"
#include "STrackInfo.h"

#include "G4Version.hh"
#include "U4Track.h"


struct U4TrackTest
{
    static void  SetTag(G4Track* track, G4int tag);
    static G4int GetTag(const G4Track* track);

    static int STrackInfo__Set();
    static int Tag();
    static int Main();
};


inline void U4TrackTest::SetTag(G4Track* track, G4int tag)
{
#if defined(G4VERSION_NUMBER) && G4VERSION_NUMBER >= 1100
    track->SetCreatorModelID(tag);
#else
    track->SetCreatorModelIndex(tag);
#endif
}

inline G4int U4TrackTest::GetTag(const G4Track* track)
{
#if defined(G4VERSION_NUMBER) && G4VERSION_NUMBER >= 1100
    return track->GetCreatorModelID();
#else
    // In Geant4 10.x, GetCreatorModelID() exists and returns fCreatorModelIndex
    return track->GetCreatorModelID();
#endif
}
inline int U4TrackTest::Tag()
{
    G4Track* track = U4Track::MakePhoton();

    int rc = 0 ;
    for(int i = -1'000'000 ; i < 1'000'000 ; i++)
    {
        G4int tag_0 = i ;
        SetTag(track, tag_0);

        G4int tag_1 = GetTag(track); ;
        rc += ( tag_0 == tag_1 ) ? 0 : 1 ;
    }

    return rc ;
}







inline int U4TrackTest::STrackInfo__Set()
{
    G4Track* track = U4Track::MakePhoton();
    spho p0 = {1, 2, 3, {0,0,0,0}} ;

    STrackInfo::Set(track, p0 );

    const G4Track* ctrack = track ;
    std::cout << U4Track::Desc(ctrack) << std::endl ;

    spho* p2 = STrackInfo::GetRef(ctrack);
    assert( p2->isIdentical(p0) );
    std::cout << U4Track::Desc(ctrack) << std::endl ;

    p2->uc4.w = 'Z' ;
    std::cout << U4Track::Desc(ctrack) << std::endl ;

    return 0;
}


inline int U4TrackTest::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "Tag");
    bool ALL = 0 == strcmp(TEST, "ALL");
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"STrackInfo__Set"))   rc += STrackInfo__Set();
    if(ALL||0==strcmp(TEST,"Tag"))               rc += Tag();
    return rc ;
}


int main(){ return U4TrackTest::Main(); }
