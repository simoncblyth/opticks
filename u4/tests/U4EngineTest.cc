/**
U4EngineTest.cc
==================

::

    epsilon:tests blyth$ U4EngineTest 
     mode:0 saving engine states 
    U4Engine::SaveState /tmp/U4EngineTest/s0.conf
     label s0.conf
        0 :     0.1305
        1 :     0.6178
        2 :     0.9959
        3 :     0.4959
        4 :     0.1129
        5 :     0.2899
        6 :     0.4730
        7 :     0.8376
        8 :     0.3594
        9 :     0.9269
    U4Engine::SaveState /tmp/U4EngineTest/s1.conf
     label s1.conf
        0 :     0.5480
        1 :     0.1372
        2 :     0.1190
        3 :     0.6559
        4 :     0.0721
        5 :     0.9804
        6 :     0.7405
        7 :     0.9630
        8 :     0.0069
        9 :     0.3705
    U4Engine::SaveState /tmp/U4EngineTest/s2.conf
     label s2.conf
        0 :     0.5728
        1 :     0.6244
        2 :     0.4719
        3 :     0.2626
        4 :     0.5345
        5 :     0.9096
        6 :     0.8778
        7 :     0.6699
        8 :     0.0987
        9 :     0.3317
    U4Engine::Desc engine Y engine.name MixMaxRng
    epsilon:tests blyth$ 
    epsilon:tests blyth$ U4EngineTest 1
     mode:1 restore engine state 
    U4Engine::RestoreState /tmp/U4EngineTest/s2.conf
     label s2.conf Restored
        0 :     0.5728
        1 :     0.6244
        2 :     0.4719
        3 :     0.2626
        4 :     0.5345
        5 :     0.9096
        6 :     0.8778
        7 :     0.6699
        8 :     0.0987
        9 :     0.3317
    U4Engine::Desc engine Y engine.name MixMaxRng



    epsilon:tests blyth$ U4EngineTest 2
     mode:2 restore engine state from array 
     states (10, 38, )
     label U4Engine::RestoreState(states, 2)
        0 :     0.5728
        1 :     0.6244
        2 :     0.4719
        3 :     0.2626
        4 :     0.5345
        5 :     0.9096
        6 :     0.8778
        7 :     0.6699
        8 :     0.0987
        9 :     0.3317
    U4Engine::Desc engine Y engine.name MixMaxRng




**/



#include <iostream>
#include "ssys.h"
#include "spath.h"
#include "NP.hh"
#include "U4Engine.h"
#include "G4Types.hh"
#include "U4UniformRand.h"

#include "Randomize.hh"

const char* FOLD = spath::Resolve("$TMP/U4EngineTest" ); 

void dump(int idx, double u)
{
    std::cout 
        << std::setw(5) << idx 
        << " : " 
        << std::fixed << std::setw(10) << std::setprecision(4) << u 
        << std::endl 
        ; 
}

void dump(const char* label, int n)
{
    std::cout << " label " << label << std::endl ; 
    for(int i=0 ; i < n ; i++) dump(i, G4UniformRand() ) ; 
}

void test_SaveState_RestoreState(int argc, char** argv)
{
    int mode = argc > 1 ? std::atoi(argv[1]) : 0 ; 
    int N = 10 ; 
    bool show = false ; 

    if( mode == 0 )
    {
        NP* states = NP::Make<unsigned long>(10, 2*17+4 ) ; 
        std::cout << " mode:0 saving engine states " << std::endl ; 
        U4Engine::SaveState(FOLD, "s0.conf") ;   
        U4Engine::SaveState( states, 0 ); 
        if(show) U4Engine::ShowState(); 
        dump("s0.conf", N ); 

        U4Engine::SaveState(FOLD, "s1.conf") ;   
        U4Engine::SaveState( states, 1 ); 
        if(show) U4Engine::ShowState(); 
        dump("s1.conf", N); 

        U4Engine::SaveState(FOLD, "s2.conf") ;   
        U4Engine::SaveState( states, 2 ); 
        if(show) U4Engine::ShowState(); 
        dump("s2.conf", N); 

        states->save(FOLD, "states.npy"); 
    }
    else if( mode == 1 )
    {
        std::cout << " mode:1 restore engine state from file " << std::endl ; 
        U4Engine::RestoreState(FOLD, "s2.conf") ;   
        if(show) U4Engine::ShowState(); 
        dump("U4Engine::RestoreState(FOLD, \"s2.conf\")", N ); 
    }
    else if( mode == 2 )
    {
        std::cout << " mode:2 restore engine state from array " << std::endl ; 
        NP* states = NP::Load(FOLD, "states.npy"); 
        std::cout << " states " << states->sstr() << std::endl; 
        U4Engine::RestoreState( states, 2 ); 
        if(show) U4Engine::ShowState(); 
        dump("U4Engine::RestoreState(states, 2)", N ); 
    }

    std::cout << U4Engine::Desc() << std::endl ; 
}


void test_Desc()
{
    std::cout << U4Engine::DescState() << std::endl ; 
    std::cout << U4Engine::DescStateArray() << std::endl ; 
}

void test_U4UniformRand()
{

    NP* states = NP::Make<unsigned long>(10, 2*17+4 ) ; 

    U4Engine::SaveState( states, 0 ); 
    NP* u0 = U4UniformRand::Get(100); 

    U4Engine::RestoreState( states, 0 ); 
    NP* u1 = U4UniformRand::Get(100); 

    U4Engine::RestoreState( states, 0 ); 
    NP* u2 = U4UniformRand::Get(100); 

    std::cout << "u0 " << u0->repr<double>() << std::endl; 
    std::cout << "u1 " << u1->repr<double>() << std::endl; 
    std::cout << "u2 " << u2->repr<double>() << std::endl; 

    u0->save(FOLD, "test_U4UniformRand", "u0.npy"); 
    u1->save(FOLD, "test_U4UniformRand", "u1.npy"); 
    u2->save(FOLD, "test_U4UniformRand", "u2.npy"); 
}


int main(int argc, char** argv)
{
    /*
    test_SaveState_RestoreState(argc, argv); 
    test_Desc(); 
    */

    test_U4UniformRand();     

    return 0 ; 
}
