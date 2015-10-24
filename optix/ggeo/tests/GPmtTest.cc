#include "GPmt.hh"
#include "GBuffer.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

int main(int argc, char* argv[])
{
    for(unsigned int i=0 ; i < argc ; i++) LOG(info) << i << ":" << argv[i] ; 
 
    const char* slice = argc > 1 ? argv[1] : NULL ;  

    GBuffer* orig = GBuffer::load<float>("/tmp/hemi-pmt-parts.npy");


    unsigned int nelem = orig->getNumElements();
    assert(nelem == 4 && "expecting quads");
    orig->reshape(4*GPmt::QUADS_PER_ITEM);  
    GBuffer* pbuf = orig->make_slice(slice);
    orig->reshape(nelem);
    pbuf->reshape(nelem);


    GPmt* pmt = new GPmt(pbuf);
    pmt->dump();
    pmt->Summary();

    GBuffer* sb = pmt->getSolidBuffer();
    sb->save<unsigned int>("/tmp/hemi-pmt-solids.npy");
    sb->dump<unsigned int>("solidBuffer partOffset/numParts/solidIndex/0 ");

    return 0 ;
}


