// op --tindexer

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "Indexer.hh"
#include "BLog.hh"

int main(int argc, char** argv)
{
    NLog nl("IndexerTest.log","info");
    nl.configure(argc, argv, "/tmp");

    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    OpticksEvent* evt = OpticksEvent::load(typ, tag, det, cat) ;
    assert(evt);   
    LOG(info) << evt->getShapeString() ; 

    NPY<unsigned long long>* sequence = evt->getSequenceData();
    NPY<unsigned char>*        phosel = evt->getPhoselData();
    assert(sequence->getShape(0) == phosel->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence(Opticks::SEQHIS_NAME_, Opticks::SEQMAT_NAME_);
    idx->applyLookup<unsigned char>(phosel->getValues());

    phosel->save("/tmp/phosel.npy"); 


    return 0 ; 
}
