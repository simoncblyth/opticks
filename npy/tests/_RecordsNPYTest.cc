
#include <cstdlib>
#include <cassert>

#include "NPY.hpp"
#include "Types.hpp"
#include "RecordsNPY.hpp"
#include "RecordsNPY.hpp"

#include "OPTICKS_LOG.hh"


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    Types types ; 
    types.dumpFlags();

    const char* pfx = "tboolean-proxy-11" ;
    const char* typ = "torch" ;
    const char* tag = "1" ;
    const char* det = pfx ;

    NPY<float>* photons = NPY<float>::load(pfx,"ox", typ, tag, det);
    if(!photons) return 0 ;   

    NPY<short>* records = NPY<short>::load(pfx,"rx", typ, tag, det);
    if(!records) return 1 ; 

    NPY<float>* fdom  = NPY<float>::load(pfx,"fdom", typ, tag, det);
    if(!fdom) return 1 ; 

    NPY<int>*   idom  = NPY<int>::load(pfx,"idom", typ, tag, det);

    if(idom == NULL)
    {  
       LOG(warning) << "FAILED TO LOAD idom " ;
    }

    unsigned int maxrec = idom ? idom->getValue(0,0,3) : 0 ;  // TODO: enumerate the k indices 
    
    if(maxrec != 10)
    {
       LOG(fatal) << "UNEXPECTED maxrec " << maxrec ;   
    }
    assert(maxrec == 10);


   
/*
    const char* idpath = getenv("IDPATH");
    types.readMaterials(idpath, "GMaterialLib");
    types.dumpMaterials();

    bool flat = true ; 
    RecordsNPY r(records, maxrec, flat);
    r.setTypes(&types);
    r.setDomains(domains);


    NPY<unsigned long long>* seqn = r.makeSequenceArray(Types::HISTORY);
    seqn->save("$TMP/seqn.npy");


     
    LOG(info) << "photon loop " ; 

    for(unsigned int i=0 ; i < 100 ; i++)
    { 
        unsigned int photon_id = i ;  
        std::string seqhis = r.getSequenceString(photon_id, Types::HISTORY );
        std::string dseqhis = types.decodeSequenceString(seqhis, Types::HISTORY);
        unsigned long long cseq = types.convertSequenceString(seqhis, Types::HISTORY);
        unsigned long long hseq = r.getSequence(photon_id, Types::HISTORY );

        printf(" photon_id %8d hseq [%16llx] cseq[%16llx] seqhis [%20s] dseqhis [%s]  \n", photon_id, hseq, cseq, seqhis.c_str(), dseqhis.c_str() );

        assert(cseq == hseq);

        //std::string seqmat = r.getSequenceString(photon_id, Types::MATERIAL );
        //std::string dseqmat = r.decodeSequenceString(seqmat, Types::MATERIAL);
        //printf(" photon_id %8d seqmat [%s] dseqmat [%s] \n", photon_id, seqmat.c_str(), dseqmat.c_str() );
    }
*/

    return 0 ;
}

