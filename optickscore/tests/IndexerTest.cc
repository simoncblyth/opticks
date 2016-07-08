// op --tindexer


#include "NGLM.hpp"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "Indexer.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    OpticksEvent* evt = OpticksEvent::load(typ, tag, det, cat) ;
    if(!evt) return 0 ; 

    LOG(info) << evt->getShapeString() ; 

    NPY<unsigned long long>* sequence = evt->getSequenceData();
    NPY<unsigned char>*        phosel = evt->getPhoselData();
    assert(sequence->getShape(0) == phosel->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence(OpticksConst::SEQHIS_NAME_, OpticksConst::SEQMAT_NAME_);
    idx->applyLookup<unsigned char>(phosel->getValues());

    phosel->save("$TMP/phosel.npy"); 


    return 0 ; 
}
