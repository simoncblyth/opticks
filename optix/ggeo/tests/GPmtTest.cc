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
    const char* slice_ = argc > 1 ? argv[1] : NULL ;  
    NSlice* slice = new NSlice(slice_);

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


