/**
OpticksEventCompareTest
========================

**/

#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"

#include "NCSGList.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventCompare.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    OpticksEvent* evt = ok.loadEvent();

    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }

    OpticksEvent* g4evt = ok.loadEvent(false);

    if(!g4evt || g4evt->isNoLoad())
    {
        LOG(fatal) << "failed to load g4evt " ; 
        return 0 ; 
    }

    OpticksEventCompare cf(evt,g4evt);
    cf.dump("cf(evt,g4evt)");

    return 0 ; 
}
