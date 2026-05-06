/**
SOpticksClientSimulator_test.cc
=================================

This is the standalone test, which uses SEvtMock.
See also the CMake built SOpticksClientSimulatorTest.cc which
should be kept very similar.

**/


#include "SOpticksClientSimulator.h"

int main()
{
#ifdef WITH_SEVT_MOCK
    SEvtMock* sev = SEvtMock::Create(0);
    sev->load_genstep("$FOLD/gs.npy");
#else
    SEvt::Create(0);
#endif

    SOpticksClientSimulator* client = SOpticksClientSimulator::Create("$CFBaseFromGEOM/CSGFoundry/SSim"); ;
    if(!client) return 1 ;
    std::cout << " client.desc:[" << client->desc() << "]\n" ;

    for(int i=0 ; i < 10 ; i++) client->simulate(i, false );

    return 0 ;
}


