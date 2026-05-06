/**
SOpticksClientSimulatorTest.cc
===============================

This is CMake built integrated test which uses SEvt.
See also the script built standalone SOpticksClientSimulator_test.cc
which should be kept very similar to this.

**/

#include "SOpticksClientSimulator.h"

int main()
{
    SEvt::Create(0);

    SOpticksClientSimulator* client = SOpticksClientSimulator::Create("$CFBaseFromGEOM/CSGFoundry/SSim"); ;
    if(!client)
    {
        std::cerr << "SOpticksClientSimulatorTest - FAILED TO CREATE CLIENT \n";
        return 1 ;
    }
    std::cout << " client.desc:[" << client->desc() << "]\n" ;

    for(int i=0 ; i < 10 ; i++)
    {
        double dt = client->simulate(i, false );
        if(dt < 0.) return 1 ;
    }

    return 0 ;
}


