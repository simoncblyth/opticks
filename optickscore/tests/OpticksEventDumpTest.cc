/**
OpticksEventDumpTest
======================


**/

#include "OKCORE_LOG.hh"
#include "NPY_LOG.hh"


#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    bool g4 = ok.hasOpt("vizg4|evtg4") ;
    OpticksEvent* evt = ok.loadEvent(!g4);

    if(!evt || evt->isNoLoad())
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }

    OpticksEventDump dump(evt);

    dump.dump(0);

    return 0 ; 
}
