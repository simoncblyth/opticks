/**
CSGOptiXServiceTest
======================

The actual FastAPI connected service needs to be
in python using a binding to the very high level
CSGOptiXService API. So this test is for refining the
very high level API before trying to nanobind to it.

Note the similarity to the embedded G4CXOpticks usage
from within JUNOSW can use techniques from there like
the embedded logging

**/


#include "OPTICKS_LOG.hh"
#include "CSGOptiXService.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CSGOptiXService cxs ;

    NP* gs = NP::Make<float>(1,6,4);  gs->fillIndexFlat();
    std::cout << "gs: " << ( gs ? gs->sstr() : "-" ) << "\n" ;

    NP* ht = cxs.simulate(gs);
    std::cout << "ht: " << ( ht ? ht->sstr() : "-" ) << "\n" ;

    return 0 ;
}
