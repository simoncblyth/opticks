//  ggv --pmt
//  ggv --pmt 0:10
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

    PLOG_(argc, argv);
    GGEO_LOG_ ;

    Opticks* ok = new Opticks(argc, argv);

    for(int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 

    NSlice* slice = ok->getAnalyticPMTSlice();
    unsigned apmtidx = ok->getAnalyticPMTIndex();


    GBndLib* blib = GBndLib::load(ok, true);

    GPmt* pmt = GPmt::load(ok, blib, apmtidx, slice);
    if(!pmt)
    {
        LOG(fatal) << argv[0] << " FAILED TO LOAD PMT " ; 
        return 0 ;
    }

    GParts* ppmt = pmt->getParts();

    NPY<float>* pb = ppmt->getPartBuffer();
    LOG(info) << "parts shape: " << pb->getShapeString() ;
    assert( pb->getDimensions() == 3 );

    ppmt->dump();
    ppmt->Summary();

    //NPY<unsigned int>* sb = ppmt->getSolidBuffer();
    //sb->dump("solidBuffer partOffset/numParts/solidIndex/0 ");

    GCSG* csg = pmt->getCSG();
    NPY<float>* cb = csg->getCSGBuffer();
    cb->dump("CSG Buffer");
    LOG(info) << "CSG Buffer shape: " << cb->getShapeString() ;

    csg->dump();


     
    GMergedMesh* mm = csg->makeMergedMesh();
    //assert(mm);
    //mm->dump();


    return 0 ;
}


