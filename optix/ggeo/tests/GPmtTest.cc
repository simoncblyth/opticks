//  ggv --pmt
//  ggv --pmt 0:10
//

#include "Opticks.hh"

#include "GCache.hh"
#include "GBndLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv, "pmttest.log");

    GCache* cache = new GCache(opticks);

    for(unsigned int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 
    NSlice* slice = argc > 1 ? new NSlice(argv[1]) : NULL ;

    GBndLib* blib = GBndLib::load(cache, true);

    GPmt* pmt = GPmt::load(cache, blib, 0, slice);

    GParts* ppmt = pmt->getParts();

    NPY<float>* pb = ppmt->getPartBuffer();
    LOG(info) << "parts shape: " << pb->getShapeString() ;
    assert( pb->getDimensions() == 3 );

    ppmt->dump();
    ppmt->Summary();

    NPY<unsigned int>* sb = ppmt->getSolidBuffer();
    sb->dump("solidBuffer partOffset/numParts/solidIndex/0 ");

    return 0 ;
}


