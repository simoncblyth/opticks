//  ggv --pmt

#include "GCache.hh"
#include "GPmt.hh"
//#include "GBuffer.hh"
#include "NPY.hpp"
#include "NLog.hpp"

int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "pmttest.log", "info");
    cache->configure(argc, argv);

    for(unsigned int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 
    const char* slice = argc > 1 ? argv[1] : NULL ;  
    
    GPmt* pmt = GPmt::load(cache, 0);

    //GBuffer* orig = pmt->getPartBuffer();

    NPY<float>* orig = pmt->getPartBuffer();

    LOG(info) << "partBuffer shape: " << orig->getShapeString() ;


    assert( orig->getDimensions() == 3 );

    //unsigned int nelem = orig->getNumElements();
    unsigned int nelem = orig->getShape(2);
    assert(nelem == 4 && "expecting quads");

    /*
    // TODO: is the below just needed for slicing ?
    orig->reshape(4*GPmt::QUADS_PER_ITEM);  
    GBuffer* pbuf = orig->make_slice(slice);
    orig->reshape(nelem);
    pbuf->reshape(nelem);
    GPmt* pmt = new GPmt(pbuf);
    */

    pmt->dump();
    pmt->Summary();

    //GBuffer* sb = pmt->getSolidBuffer();
    NPY<unsigned int>* sb = pmt->getSolidBuffer();

    //sb->save<unsigned int>("/tmp/hemi-pmt-solids.npy");
    //sb->dump<unsigned int>("solidBuffer partOffset/numParts/solidIndex/0 ");

    return 0 ;
}


