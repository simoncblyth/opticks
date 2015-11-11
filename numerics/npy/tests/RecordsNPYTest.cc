#include "NPY.hpp"
#include "Types.hpp"
#include "RecordsNPY.hpp"
#include "RecordsNPY.hpp"

#include "stdlib.h"
#include "assert.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<short>* records = NPY<short>::load("rxcerenkov", tag, "dayabay");
    NPY<float>* domains = NPY<float>::load("domain","1","dayabay");
    NPY<int>*   idom = NPY<int>::load("idomain","1","dayabay");

    unsigned int maxrec = idom->getValue(0,0,3) ; // TODO: enumerate the k indices 
    assert(maxrec == 10);


    Types types ; 
    types.readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");
    types.dumpFlags();
    types.readMaterials(idpath, "GMaterialIndex");
    types.dumpMaterials();

    RecordsNPY r(records, maxrec);
    r.setTypes(&types);
    r.setDomains(domains);


    NPY<unsigned long long>* seqn = r.makeSequenceArray(Types::HISTORY);
    seqn->save("/tmp/seqn.npy");

    for(unsigned int i=0 ; i < 100 ; i++)
    { 
        unsigned int photon_id = i ;  
        std::string seqhis = r.getSequenceString(photon_id, Types::HISTORY );
        std::string dseqhis = types.decodeSequenceString(seqhis, Types::HISTORY);
        unsigned long long cseq = types.convertSequenceString(seqhis, Types::HISTORY);
        unsigned long long hseq = r.getSequence(photon_id, Types::HISTORY );
        assert(cseq == hseq);

        printf(" photon_id %8d hseq [%16llx] cseq[%16llx] seqhis [%20s] dseqhis [%s]  \n", photon_id, hseq, cseq, seqhis.c_str(), dseqhis.c_str() );

        //std::string seqmat = r.getSequenceString(photon_id, Types::MATERIAL );
        //std::string dseqmat = r.decodeSequenceString(seqmat, Types::MATERIAL);
        //printf(" photon_id %8d seqmat [%s] dseqmat [%s] \n", photon_id, seqmat.c_str(), dseqmat.c_str() );
    }


    return 0 ;
}

