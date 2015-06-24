#include "NPY.hpp"
#include "Types.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "GLMPrint.hpp"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<float>* photons = NPY<float>::load("oxcerenkov", tag);
    NPY<short>* records = NPY<short>::load("rxcerenkov", tag);
    NPY<float>* domains = NPY<float>::load("domain","1");

    unsigned int maxrec = 10 ;  // hmm find a slot in domain for such ints ?


    Types types ; 
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    types.dumpFlags();
    types.readMaterials(idpath, "GMaterialIndexLocal.json");
    types.dumpMaterials();

    RecordsNPY r(records, maxrec);
    r.setTypes(&types);
    r.setDomains(domains);

    PhotonsNPY p(photons);
    p.setTypes(&types);
    p.setRecs(&r);



    p.dump("oxc.dump");

    p.classify();
    p.classify(true);

    p.examinePhotonHistories();
    //pn.dumpRecords("records", 30);
    p.dumpPhotons("photons", 30);

    p.prepSequenceIndex();

    NPY<unsigned char>* seqidx = p.getSeqIdx();
    seqidx->setVerbose();
    seqidx->save("seqidx", tag);

    return 0 ;
}
