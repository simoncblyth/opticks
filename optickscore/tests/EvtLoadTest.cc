// op --tevtload

#include "OpticksEvent.hh"
#include "Indexer.hh"
#include "BLog.hh"

int main(int argc, char** argv)
{
    NLog nl("EvtLoadTest.log","info");
    nl.configure(argc, argv, "/tmp");

    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    OpticksEvent* evt = OpticksEvent::load(typ, tag, det, cat) ;
    assert(evt);   

    LOG(info) << evt->getShapeString() ; 


    return 0 ; 
}
