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

#include "GGEO_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ; 
    GGEO_LOG__ ; 
    OKCORE_LOG__ ; 

    LOG(info) << argv[0] ; 


    Opticks ok(argc, argv);

    ok.configure() ; // hub not available at ggeo- level 

    Types* types = ok.getTypes();
    GBndLib* blib = GBndLib::load(&ok, true); 
    GMaterialLib* mlib = blib->getMaterialLib();
    //GSurfaceLib*  slib = blib->getSurfaceLib();

    // see GGeo::setupTyp
    Typ* typ = ok.getTyp();
    typ->setMaterialNames(mlib->getNamesMap());

    //OpticksFlags* flags = ok->getFlags();
    //typ->setFlagNames(flags->getNamesMap());
    typ->setFlagNames(ok.getFlagNamesMap());


    OpticksEvent* evt = ok.loadEvent();
    if(evt->isNoLoad())
    {
        LOG(error) << "failed to load evt from " << evt->getDir()  ;
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
    RecordsNPY* rec = new RecordsNPY(rx, maxrec);
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
