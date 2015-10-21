#include "GCache.hh"
#include "GBuffer.hh"

/*
   ggv --gbuffer
*/

int main()
{
    GCache gc("GGEOVIEW_");
    GBuffer* buf = GBuffer::load<float>(gc.getIdPath(), "GMergedMesh/1/itransforms.npy" );
    buf->Summary();
    buf->dump("itran", 16);

    GBuffer* sbuf = buf->make_slice("660:672") ;
    sbuf->dump("sbuf", 16 );

    return 0 ; 
}
