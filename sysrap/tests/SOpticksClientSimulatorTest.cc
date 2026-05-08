/**
SOpticksClientSimulatorTest.cc
===============================

As this test needs the server to be running the failure
to create the client due to CFBaseFromGEOM resolution failing
does not cause a test failure in ordinary ctest running.

Test with::

    ~/o/sysrap/tests/SOpticksClientSimulatorTest.sh

**/

#include "SOpticksClientSimulator.h"

int main()
{
    SEvt::Create(0);

    SOpticksClientSimulator* client = SOpticksClientSimulator::Create("$CFBaseFromGEOM/CSGFoundry/SSim"); ;
    if(!client)
    {
        std::cerr << "SOpticksClientSimulatorTest - FAILED TO CREATE CLIENT \n";
        return 0 ; // easy out for ctest running without geom env setup
    }
    std::cout << " client.desc:[" << client->desc() << "]\n" ;

    for(int i=0 ; i < 10 ; i++)
    {
        double dt = client->simulate(i, false );
        if(dt < 0.) return 1 ;
    }

    return 0 ;
}


