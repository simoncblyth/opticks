//  op --pmt
//  op --pmt 0:10
//

#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

#include "Opticks.hh"

#include "GBndLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "GCSG.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"
#include "GGEO_BODY.hh"

int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);
    GGEO_LOG__ ;

    Opticks* ok = new Opticks(argc, argv);

    for(int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 

    NSlice* slice = ok->getAnalyticPMTSlice();
    unsigned apmtidx = ok->getAnalyticPMTIndex();

    GBndLib* blib = GBndLib::load(ok, true);

    GPmt* pmt = GPmt::load(ok, blib, apmtidx, slice);

    LOG(info) << argv[0] << " apmtidx " << apmtidx << " pmt " << pmt ; 
    if(!pmt)
    {
        LOG(fatal) << argv[0] << " FAILED TO LOAD PMT " ; 
        return 0 ;
    }

    GParts* ppmt = pmt->getParts();

    const char* containing_material = "MineralOil" ; 
    ppmt->setContainingMaterial(containing_material);
    ppmt->close(); // registerBoundaries, makePrimBuffer


    NPY<float>* pb = ppmt->getPartBuffer();
    LOG(info) << "parts shape: " << pb->getShapeString() ;
    assert( pb->getDimensions() == 3 );

    LOG(fatal) << "GParts.ppmt->dump()" ; 
    ppmt->dump();

    LOG(fatal) << "GParts.ppmt->Summary()" ; 
    ppmt->Summary();

    //NPY<unsigned int>* sb = ppmt->getSolidBuffer();
    //sb->dump("solidBuffer partOffset/numParts/solidIndex/0 ");

    GCSG* csg = pmt->getCSG();
    NPY<float>* cb = csg->getCSGBuffer();

    LOG(fatal) << "NPY.cb->dump()" ; 
    cb->dump("CSG Buffer");
    LOG(info) << "CSG Buffer shape: " << cb->getShapeString() ;
    LOG(fatal) << "GCSG.csg->dump()" ; 
    csg->dump();
   
    //GMergedMesh* mm = csg->makeMergedMesh();
    //assert(mm);
    //mm->dump();

    return 0 ;
}


