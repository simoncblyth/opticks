#include "Opticks.hh"
#include "GBuffer.hh"

/*
   ggv --gbuffer
*/


/*
void test_slice(Opticks& ok)
{
   // TODO: update: itransforms have been migrated to NPY and use 3d eg (672, 4, 4), see GMesh

    GBuffer* buf = GBuffer::load<float>(ok.getIdPath(), "GMergedMesh/1/itransforms.npy" );
    buf->Summary();
    buf->dump<float>("itran");

    GBuffer* sbuf = buf->make_slice("660:672") ;
    sbuf->reshape(4);
    sbuf->dump<float>("sbuf");
}
*/


void test_reshape(Opticks& ok)
{
    GBuffer* buf = GBuffer::load<int>(ok.getIdPath(), "GMergedMesh/1/indices.npy" );
    buf->Summary();
    buf->dump<int>("indices", 50);

    buf->reshape(3);
    buf->dump<int>("indices after reshape(3)", 50);

    buf->reshape(1);
    buf->dump<int>("indices after reshape(1)", 50);
}

void test_reshape_slice(Opticks& ok)
{
    GBuffer* buf = GBuffer::load<int>(ok.getIdPath(), "GMergedMesh/1/indices.npy" );
    buf->Summary();

    unsigned int nelem = buf->getNumElements();
    buf->reshape(3);
    GBuffer* sbuf = buf->make_slice("0:4") ;
    buf->reshape(nelem);     // return to original 

    sbuf->dump<int>("reshape(3) buffer sliced with 0:4",100);

    sbuf->reshape(1);
    sbuf->dump<int>("after reshape(1)",100);
}



int main()
{
    Opticks ok ; 

    //test_slice(gc);
    test_reshape(ok);
    test_reshape_slice(ok);

    return 0 ; 
}
