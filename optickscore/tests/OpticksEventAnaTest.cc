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
    PLOG_COLOR(argc, argv);

    NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    OpticksEvent* evt = ok.loadEvent(true);
    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load ok evt " ; 
        return 0 ; 
    }
    const char* geopath = evt->getGeoPath();
    LOG(info) << " geopath : " << ( geopath ? geopath : "-" ) ; 

    NCSGList* csglist = NCSGList::Load(geopath, ok.getVerbosity() );
    csglist->dump();

    OpticksEventAna*  okana = new OpticksEventAna(&ok, evt, csglist);
    okana->dump("GGeoTest::anaEvent.ok");
   

    OpticksEvent* g4evt = ok.loadEvent(false);
    if(!g4evt || g4evt->isNoLoad())
    {
        LOG(fatal) << "failed to load g4 evt " ; 
        return 0 ; 
    }
 
    const char* geopath2 = g4evt->getGeoPath();
    assert( strcmp( geopath, geopath2) == 0 );

    OpticksEventAna* g4ana = new OpticksEventAna(&ok, g4evt, csglist);
    g4ana->dump("GGeoTest::anaEvent.g4");




    if(okana) okana->dumpPointExcursions("ok");
    if(g4ana) g4ana->dumpPointExcursions("g4");
    
   

    return 0 ; 
}
