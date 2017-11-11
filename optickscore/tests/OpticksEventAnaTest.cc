/**
OpticksEventAnaTest
=====================

Pulling together an evt and the NCSGList geometry 
it came from, for intersect tests.


**/

#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"

#include "NCSGList.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"
#include "OpticksEventAna.hh"

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

    const char* geopath = evt->getGeoPath();
    LOG(info) 
              << " geopath : " << ( geopath ? geopath : "-" )
               ; 

    NCSGList* csglist = NCSGList::Load(geopath, ok.getVerbosity() );
    csglist->dump();

    OpticksEventAna ana(&ok, evt, csglist);
    ana.dump("GGeoTest::anaEvent");

    return 0 ; 
}
