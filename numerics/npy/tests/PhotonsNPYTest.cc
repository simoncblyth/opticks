#include "NPY.hpp"
#include "PhotonsNPY.hpp"
#include "GLMPrint.hpp"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* tag = "1" ;

    NPY<float>* photons = NPY<float>::load("oxcerenkov", tag);
    NPY<short>* records = NPY<short>::load("rxcerenkov", tag);
    NPY<float>* domain = NPY<float>::load("domain","1");
    domain->dump();

    glm::vec4 ce = domain->getQuad(0,0);
    glm::vec4 td = domain->getQuad(1,0);
    glm::vec4 wd = domain->getQuad(2,0);
    print(ce, "ce");
    print(td, "td");
    print(wd, "wd");

    unsigned int maxrec = 10 ;  // hmm find a slot in domain for such ints ?

    PhotonsNPY pn(photons, records, maxrec);

    pn.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    pn.dumpFlags();

    pn.readMaterials(idpath, "GMaterialIndexLocal.json");
    pn.dumpMaterials();
    pn.setCenterExtent(ce);    
    pn.setTimeDomain(td);    
    pn.setWavelengthDomain(wd);    


    pn.dump("oxc.dump");

    pn.classify();
    pn.classify(true);

    pn.examinePhotonHistories();
    //pn.dumpRecords("records", 30);
    pn.dumpPhotons("photons", 30);

    pn.prepSequenceIndex();

    NPY<unsigned char>* seqidx = pn.getSeqIdx();
    seqidx->save("seqidx", tag);

    return 0 ;
}
