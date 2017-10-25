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
    //NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    OpticksEvent* evt = ok.loadEvent();

    if(!evt)
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }

    const char* dbgcsgpath = ok.getDbgCSGPath();
    int dbgnode = ok.getDbgNode(); 

    if(!dbgcsgpath)
    {
         LOG(fatal) << " missing --dbgcsgpath " ; 
         return 0 ;  
    }

    LOG(info) << " dbgcsgpath : " << dbgcsgpath 
              << " dbgnode : " << dbgnode  
               ; 

    NCSGList csglist(dbgcsgpath, ok.getVerbosity() );
    csglist.dump();

    OpticksEventAna ana(&ok, evt, &csglist);
    ana.dump("GGeoTest::anaEvent");

    return 0 ; 
}
