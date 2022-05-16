// name=sgs_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include "sgs.h"
#include <iostream>


void test_assign()
{
    sgs gs ; 

    gs = {} ; 
    std::cout << gs.desc() << std::endl ; 

    gs = { 101, 10000, 5000, OpticksGenstep_TORCH } ; 
    std::cout << gs.desc() << std::endl ; 

    gs = { 202, 20000, 4000, OpticksGenstep_CERENKOV } ; 
    std::cout << gs.desc() << std::endl ; 

    gs = { 303, 30000, 3000, OpticksGenstep_SCINTILLATION } ; 
    std::cout << gs.desc() << std::endl ; 

    gs = { 303, 30000, 3000, OpticksGenstep_G4Cerenkov_1042 } ; 
    std::cout << gs.desc() << std::endl ; 
}

void test_array()
{
    sgs gs[] = { 
        { 101, 10000, 5000, OpticksGenstep_TORCH },
        { 202, 20000, 4000, OpticksGenstep_CERENKOV },
        { 303, 30000, 3000, OpticksGenstep_SCINTILLATION },
        { 303, 30000, 3000, OpticksGenstep_G4Cerenkov_1042 } 
        }; 

    std::cout 
         << " sizeof(sgs) " << sizeof(sgs) 
         << " sizeof(gs) " << sizeof(gs)
         << std::endl
         ; 

    for(int i=0 ; i < sizeof(gs)/sizeof(sgs) ; i++) std::cout << gs[i].desc() << std::endl ;  
}

void test_MakePho_reemit()
{

    int index = 0 ; 
    int photons = 10 ; 
    int offset = 1000 ; 
    int gentype = OpticksGenstep_SCINTILLATION ; 

    sgs gs = { index, photons, offset, gentype } ; 

    std::cout << " gs.desc " << gs.desc() << std::endl ; 

    spho non = spho::Placeholder(); 

    spho p1[gs.photons] ; 
    spho p2[gs.photons] ; 
    spho p3[gs.photons] ; 

    std::cout << "first loop with ancestor placeholder " << std::endl ; 
    for(int idx=0 ; idx < gs.photons ; idx++) p1[idx] = gs.MakePho(idx, non) ; 
    for(int idx=0 ; idx <  gs.photons ; idx++) std::cout << " p1.desc " << p1[idx].desc() << std::endl ; 

    std::cout << "second loop with ancestor to the preceeding generation " << std::endl ; 
    for(int idx=0 ; idx < gs.photons ; idx++) p2[idx] = gs.MakePho(idx, p1[idx] ) ; 
    for(int idx=0 ; idx < gs.photons ; idx++) std::cout << " p2.desc " << p2[idx].desc() << std::endl ; 

    std::cout << "third loop with ancestor to the preceeding generation " << std::endl ; 
    for(int idx=0 ; idx < gs.photons ; idx++) p3[idx] = gs.MakePho(idx, p2[idx] ) ; 
    for(int idx=0 ; idx < gs.photons ; idx++) std::cout << " p3.desc " << p3[idx].desc() << std::endl ; 

}


int main()
{
    /*
    test_assign(); 
    test_array();  
    */

    test_MakePho_reemit(); 


    return 0 ; 
}
