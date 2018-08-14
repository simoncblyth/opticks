/**
OpticksEventDumpTest
======================


**/

#include "OPTICKS_LOG.hh"

#include "BOpticksKey.hh"
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    // BOpticksKey::SetKey(NULL);  // <-- makes sensitive to OPTICKS_KEY envvar 
    // this is done internally at Opticks instanciation when have argument --envkey 

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
