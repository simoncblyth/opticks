/**
CSGOptiXServiceTest
======================

The actual FastAPI connected service needs to be
in python using a binding to the very high level
CSGOptiXService API. So this test is for refining the
very high level API that will be used from python via
nanobind.

Note the similarity to the embedded G4CXOpticks usage
from within JUNOSW can use techniques from there like
the embedded logging

**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "CSGOptiXService.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CSGOptiXService cxs ;

    // WIP: use input genstep machinery for this test
    SEvt* sev = SEvt::Get_EGPU();


    // in server-client situation the gensteps and eventID
    // will arrive in HTTP request body and headers
    int eventID = 0 ;
    sev->setIndex(eventID);
    // need to set index before creating input genstep
    // for the number of photons config to work

    NP* gs = sev->createInputGenstep_configured();
    assert( gs );
    gs->set_meta<int>("eventID", eventID );  // required by QSim::simulate


    std::cout << "gs: " << ( gs ? gs->sstr() : "-" ) << "\n" ;

    NP* ht = cxs.simulate(gs);

    std::cout << "ht: " << ( ht ? ht->sstr() : "-" ) << "\n" ;

    return 0 ;
}
