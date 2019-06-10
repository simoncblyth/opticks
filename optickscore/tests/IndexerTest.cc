// op --tindexer

/**
IndexerTest
=============

::

   ckm-indexer-test 


**/

#include "NGLM.hpp"
#include "Opticks.hh"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "Indexer.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* pfx = "source" ;  // 
    const char* typ = "natural" ; 
    const char* tag = "1" ; 
    const char* det = "g4live" ; 
    const char* cat = NULL ; 
 
    OpticksEvent* evt = OpticksEvent::load(pfx, typ, tag, det, cat) ;

    if(!evt) 
    {
        LOG(info) << " failed to load " ; 
        return 0 ; 
    }

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
