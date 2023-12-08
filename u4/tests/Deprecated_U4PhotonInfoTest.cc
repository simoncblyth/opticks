#include "Deprecated_U4PhotonInfo.h"
#include "U4Track.h"
#include <iostream>
#include <csignal>


void test_SetGet()
{
    G4Track* track = U4Track::MakePhoton(); 
    spho p0 = {1, 2, 3, {0,0,0,0}} ; 
    Deprecated_U4PhotonInfo::Set(track, p0 ); 
    const G4Track* ctrack = track ; 
    spho p1 = Deprecated_U4PhotonInfo::Get(ctrack); 
    assert( p1.isIdentical(p0) ); 

    std::cout 
        << " track " << track << std::endl 
        << " p0 " << p0.desc() << std::endl 
        << " p1 " << p1.desc() << std::endl 
        ; 
}

void test_GetRef()
{
    G4Track* track = U4Track::MakePhoton(); 
    spho p0 = {1, 2, 3, {0,0,0,0}} ; 
    Deprecated_U4PhotonInfo::Set(track, p0 ); 
    const G4Track* ctrack = track ; 
    spho p1 = Deprecated_U4PhotonInfo::Get(ctrack); 
    bool p1_expect = p1.isIdentical(p0) ;
    assert( p1_expect ); 
    if(!p1_expect) std::raise(SIGINT); 

    spho* p2 = Deprecated_U4PhotonInfo::GetRef(ctrack); 
    bool p2_expect =  p2->isIdentical(p0) ;
    assert( p2_expect ); 
    if(!p2_expect) std::raise(SIGINT); 

    std::cout << " p2 " << p2->desc() << " p2 from Deprecated_U4PhotonInfo::GetRef(ctrack) " << std::endl ; 

    p2->set_gen(10); 
    p2->set_flg(101); 
    std::cout << " p2 " << p2->desc() << " after p2->set_flg(101) p2->set_gen(10) " << std::endl ; 

    spho p3 = Deprecated_U4PhotonInfo::Get(ctrack); 
    std::cout << " p3 " << p3.desc() << " Deprecated_U4PhotonInfo::Get(ctrack)  by value reflects the change made using the pointer " << std::endl ; 


}



int main()
{
    /*
    test_SetGet(); 
    */
    test_GetRef();  
    


    return 0 ; 
}
