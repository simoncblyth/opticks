// op --tindexer

#include "NumpyEvt.hpp"
#include "Indexer.hh"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    NLog nl("IndexerTest.log","info");
    nl.configure(argc, argv, "/tmp");

    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    NumpyEvt* evt = NumpyEvt::load(typ, tag, det, cat) ;
    assert(evt);   
    LOG(info) << evt->getShapeString() ; 

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>() ; 
    idx->setEvt(evt); 
    idx->indexSequence();



    return 0 ; 
}
