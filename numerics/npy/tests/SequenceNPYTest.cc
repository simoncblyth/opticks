#include "NPY.hpp"
#include "Types.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "SequenceNPY.hpp"
#include "Index.hpp"

#include "assert.h"
#include "stdlib.h"



/*
#ifdef SLOW_CPU_INDEXING
    SequenceNPY seq(dpho);
    seq.setTypes(&types);
    seq.setRecs(&rec);
    seq.setSeqIdx(evt.getRecselData());
    seq.indexSequences(32);   // creates and populates the seqidx CPU side 

    Index* seqhis = seq.getSeqHis();
    Index* seqmat = seq.getSeqMat();
    //seqidx->save("seq%s", typ, tag);  

*/




int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<float>* photons = NPY<float>::load("oxcerenkov", tag, "dayabay");
    NPY<short>* records = NPY<short>::load("rxcerenkov", tag, "dayabay");
    NPY<float>* dom = NPY<float>::load("domain","1", "dayabay");
    NPY<int>*   idom = NPY<int>::load("idomain","1", "dayabay");
    unsigned int maxrec = idom->getValue(0,0,3) ; 
    assert(maxrec == 10);

    Types types ; 
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    types.dumpFlags();
    //types.readMaterialsOld(idpath, "GMaterialIndexLocal.json");
    types.readMaterials(idpath, "GMaterialIndex");
    types.dumpMaterials();

    bool flat = false ; 
    RecordsNPY r(records, maxrec, flat);
    r.setTypes(&types);
    r.setDomains(dom);

    SequenceNPY s(photons);
    s.setTypes(&types);
    s.setRecs(&r);

    s.countMaterials();
    s.dumpUniqueHistories();
    s.indexSequences(32);

    NPY<unsigned long long>* seqhisnpy = s.getSeqHisNpy();
    seqhisnpy->save("/tmp/SequenceNPYTest_SeqHis.npy");

    NPY<unsigned char>* seqidx = s.getSeqIdx();
    seqidx->setVerbose();
    seqidx->save("/tmp/SequenceNPYTest_SeqIdx.npy");

    return 0 ;
}

