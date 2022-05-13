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

int main()
{
    //test_assign(); 
    test_array();  
    return 0 ; 
}
