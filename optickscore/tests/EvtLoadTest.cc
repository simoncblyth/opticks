// op --tevtload
// tlens-load 

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

    bool ok_ = true ; 
    unsigned tagoffset = 0 ; 

    OpticksEvent* evt = ok.loadEvent(ok_, tagoffset);

    if(!evt)
    {
        LOG(fatal) << "failed to load evt " ; 
        return 0 ; 
    }


    OpticksEventDump dmp(evt);
    dmp.dump();



    return 0 ; 
}
