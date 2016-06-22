#include "Opticks.hh"
#include "GBuffer.hh"
#include "PLOG.hh"
#include "GGEO_LOG.hh"

/*
   ggv --gbuffer
*/


void test_reshape(Opticks& ok)
{
    const char* idpath = ok.getIdPath() ;
    LOG(info) << "[" 
              << " idpath " << idpath 
              ; 

    GBuffer* buf = GBuffer::load<int>(idpath, "GMergedMesh/1/indices.npy" );
    if(!buf) return ; 
 
    buf->Summary();
    buf->dump<int>("indices", 50);

    buf->reshape(3);
    buf->dump<int>("indices after reshape(3)", 50);

    buf->reshape(1);
    buf->dump<int>("indices after reshape(1)", 50);

    LOG(info) << "]" ; 
}

void test_reshape_slice(Opticks& ok)
{
    const char* idpath = ok.getIdPath() ;
    LOG(info) << "[" ; 
    GBuffer* buf = GBuffer::load<int>(idpath, "GMergedMesh/1/indices.npy" );
    if(!buf) return ; 
    buf->Summary();

    unsigned int nelem = buf->getNumElements();
    buf->reshape(3);
    GBuffer* sbuf = buf->make_slice("0:4") ;
    buf->reshape(nelem);     // return to original 

    sbuf->dump<int>("reshape(3) buffer sliced with 0:4",100);

    sbuf->reshape(1);
    sbuf->dump<int>("after reshape(1)",100);
    LOG(info) << "]" ; 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;  

 
    LOG(info) << argv[0] ; 

    Opticks ok ; 

    LOG(info) << " after ok " ; 

    test_reshape(ok);
    test_reshape_slice(ok);

    return 0 ; 
}
