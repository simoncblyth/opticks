#include "NPY.hpp"
#include "Types.hpp"
#include "RecordsNPY.hpp"

#include "stdlib.h"
#include "assert.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<short>* records = NPY<short>::load("rxcerenkov", tag);
    NPY<float>* domains = NPY<float>::load("domain","1");
    NPY<int>*   idom = NPY<int>::load("idomain","1");
    unsigned int maxrec = idom->getValue(0,0,3) ; // TODO: enumerate the k indices 
    assert(maxrec == 10);

    Types types ; 
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    types.dumpFlags();
    types.readMaterials(idpath, "GMaterialIndexLocal.json");
    types.dumpMaterials();

    RecordsNPY r(records, maxrec);
    r.setTypes(&types);
    r.setDomains(domains);

 
    unsigned int photon_id = 0 ; 

    std::string seqhis = r.getSequenceString(photon_id, Types::HISTORY );
    std::string dseqhis = r.decodeSequenceString(seqhis, Types::HISTORY);
    printf(" photon_id %8d seqhis [%s] dseqhis [%s] \n", photon_id, seqhis.c_str(), dseqhis.c_str() );

    std::string seqmat = r.getSequenceString(photon_id, Types::MATERIAL );
    std::string dseqmat = r.decodeSequenceString(seqmat, Types::MATERIAL);
    printf(" photon_id %8d seqmat [%s] dseqmat [%s] \n", photon_id, seqmat.c_str(), dseqmat.c_str() );



    return 0 ;
}

