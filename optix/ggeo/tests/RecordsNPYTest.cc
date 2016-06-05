//  ggv --recs

#include "Types.hpp"
#include "Typ.hpp"

#include "RecordsNPY.hpp"
#include "PhotonsNPY.hpp"

#include "Opticks.hh"

#include "OpticksFlags.cc"
#include "GBndLib.cc"
#include "GMaterialLib.cc"
#include "GSurfaceLib.cc"

// architecture problem, need ggeo-/GPropertyLib funciontality 
// regards meanings of things at the lower npy- level 
// (in python just read in from persisted)
//
// maybe retain Types and simplify it to a holder of maps
// obtained from the higher level, or create a higher level 
// GEvt ?

int main(int argc, char** argv)
{
    // canonically App::indexEvtOld , contrast with npy-/ana.py

    const char* src = "torch" ; 
    const char* tag = "2" ; 
    Opticks* cache = new Opticks(argc, argv, "recs.log");

    Types* types = cache->getTypes();

    GBndLib* blib = GBndLib::load(cache, true); 
    GMaterialLib* mlib = blib->getMaterialLib();
    //GSurfaceLib*  slib = blib->getSurfaceLib();

    // see GGeo::setupTyp
    Typ* typ = cache->getTyp();
    OpticksFlags* flags = cache->getFlags();
    typ->setMaterialNames(mlib->getNamesMap());
    typ->setFlagNames(flags->getNamesMap());


    const char* idpath = cache->getIdPath();
    NPY<float>* fdom = NPY<float>::load(idpath, "OPropagatorF.npy");
    //NPY<int>*   idom = NPY<int>::load(idpath, "OPropagatorI.npy");

    // array([[[9, 0, 0, 0]]], dtype=int32)     ??? no 10 for maxrec 
    // NumpyEvt::load to do this ?

    NPY<float>* ox = NPY<float>::load("ox", src, tag, "dayabay");
    ox->Summary();

    NPY<short>* rx = NPY<short>::load("rx", src, tag, "dayabay");
    rx->Summary();

    unsigned int maxrec = 10 ; 
    bool flat = true ; 
    RecordsNPY* rec = new RecordsNPY(rx, maxrec, flat);
    rec->setDomains(fdom);
    rec->setTypes(types);
    rec->setTyp(typ);

    PhotonsNPY* pho = new PhotonsNPY(ox);
    pho->setRecs(rec);
    pho->setTypes(types);
    pho->setTyp(typ);
    pho->dump(0  ,  "ggv --recs dpho 0");

    return 0 ; 
}
