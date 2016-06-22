//  ggv --pmt
//  ggv --pmt 0:10
//

#include "Opticks.hh"

#include "GBndLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "GCSG.hh"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv);

    for(int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 
    NSlice* slice = argc > 1 ? new NSlice(argv[1]) : NULL ;

    GBndLib* blib = GBndLib::load(opticks, true);

    GPmt* pmt = GPmt::load(opticks, blib, 0, slice);
    if(!pmt)
    {
        LOG(fatal) << argv[0] << " FAILED TO LOAD PMT " ; 
        return 1 ;
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



    return 0 ;
}


