#include "NPY.hpp"
#include "PhotonsNPY.hpp"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");

    NPY<float>* photons = NPY<float>::load("oxcerenkov", "1");
    NPY<short>* records = NPY<short>::load("rxcerenkov", "1");

    PhotonsNPY pn(photons, records);

    pn.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    pn.dumpFlags();

    pn.readMaterials(idpath, "GMaterialIndexLocal.json");
    pn.dumpMaterials();


    pn.dump("oxc.dump");

    pn.classify();
    pn.classify(true);

    pn.examineHistories(PhotonsNPY::PHOTONS);
    pn.examineHistories(PhotonsNPY::RECORDS); // dont make much sense currently need to devise record traverse
    pn.dumpRecords("records", 30);



    return 0 ;
}
