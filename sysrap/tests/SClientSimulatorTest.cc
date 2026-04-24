/**
SClientSimulatorTest.cc
========================

This is CMake built integrated test which uses SEvt.
See also the script built standalone SClientSimulator_test.cc
which should be kept very similar to this.

**/

#include "SClientSimulator.h"

int main()
{
#ifdef WITH_SEVT_MOCK
    SEvtMock::Create(0);
#else
    SEvt::Create(0);
#endif

    SClientSimulator* client = SClientSimulator::Create("$CFBaseFromGEOM/CSGFoundry/SSim"); ;
    if(!client) 
    {
        std::cerr << "SClientSimulatorTest - FAILED TO CREATE CLIENT \n";
        return 0 ;
    }
    std::cout << " client.desc:[" << client->desc() << "]\n" ;

    for(int i=0 ; i < 10 ; i++) client->simulate(i, false );

    return 0 ;
}


