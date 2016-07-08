//  ggv --recs

#include <cassert>

#include "Types.hpp"
#include "Typ.hpp"
#include "RecordsNPY.hpp"
#include "PhotonsNPY.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"


// architecture problem, need ggeo-/GPropertyLib funciontality 
// regards meanings of things at the lower npy- level 
// (in python just read in from persisted)
//
// maybe retain Types and simplify it to a holder of maps
// obtained from the higher level, or create a higher level 
// GEvt ?

int main(int argc, char** argv)
{

    PLOG_(argc, argv);


    LOG(info) << argv[0] ; 

    // canonically App::indexEvtOld , contrast with npy-/ana.py

    Opticks* ok = new Opticks(argc, argv);

    LOG(info) << " after ok " ; 

    Types* types = ok->getTypes();

    GBndLib* blib = GBndLib::load(ok, true); 
    GMaterialLib* mlib = blib->getMaterialLib();
    //GSurfaceLib*  slib = blib->getSurfaceLib();

    // see GGeo::setupTyp
    Typ* typ = ok->getTyp();
    typ->setMaterialNames(mlib->getNamesMap());

    //OpticksFlags* flags = ok->getFlags();
    //typ->setFlagNames(flags->getNamesMap());
    typ->setFlagNames(ok->getFlagNamesMap());

    const char* src = "torch" ; 
    const char* tag = "1" ; 
    const char* det = ok->getDetector() ; 

    OpticksEvent* evt = OpticksEvent::load(src, tag, det) ;
    if(!evt)
    {
        LOG(error) << "failed to load evt " ;
        return 0 ;  
    }    

    NPY<float>* fd = evt->getFDomain();
    NPY<float>* ox = evt->getPhotonData();
    NPY<short>* rx = evt->getRecordData();


/*
    const char* idpath = cache->getIdPath();
    NPY<float>* fdom = NPY<float>::load(idpath, "OPropagatorF.npy");
    //NPY<int>*   idom = NPY<int>::load(idpath, "OPropagatorI.npy");

    // array([[[9, 0, 0, 0]]], dtype=int32)     ??? no 10 for maxrec 
    // NumpyEvt::load to do this ?

    NPY<float>* ox = NPY<float>::load("ox", src, tag, "dayabay");
    ox->Summary();

    NPY<short>* rx = NPY<short>::load("rx", src, tag, "dayabay");
    rx->Summary();

*/


    unsigned int maxrec = 10 ; 
    bool flat = true ; 
    RecordsNPY* rec = new RecordsNPY(rx, maxrec, flat);
    rec->setDomains(fd);
    rec->setTypes(types);
    rec->setTyp(typ);

    PhotonsNPY* pho = new PhotonsNPY(ox);
    pho->setRecs(rec);
    pho->setTypes(types);
    pho->setTyp(typ);
    pho->dump(0  ,  "ggv --recs dpho 0");

    return 0 ; 
}
