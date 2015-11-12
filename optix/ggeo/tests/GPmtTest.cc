//  ggv --pmt
//  ggv --pmt 0:10
//

#include "GCache.hh"
#include "GPmt.hh"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "pmttest.log", "info");
    cache->configure(argc, argv);

    for(unsigned int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 
    NSlice* slice = argc > 1 ? new NSlice(argv[1]) : NULL ;

    GPmt* pmt = GPmt::load(cache, 0, slice);

    NPY<float>* parts = pmt->getPartBuffer();
    LOG(info) << "parts shape: " << parts->getShapeString() ;
    assert( parts->getDimensions() == 3 );

    pmt->dump();
    pmt->Summary();

    NPY<unsigned int>* sb = pmt->getSolidBuffer();
    sb->dump("solidBuffer partOffset/numParts/solidIndex/0 ");

    return 0 ;
}


