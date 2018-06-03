#include "NPY.hpp"
#include "YOGGeometry.hh"


void YOGGeometry::make_triangle()
{
    /*

         (0,1)
          2
          |\
          | \
          |  \
          |   \
          |    \
          0-----1     
        (0,0)  (1,0)

    */


    vtx = NPY<float>::make(3, 4) ; 
    vtx->zero();
    vtx->setQuad(0,0,0,    0.f, 0.f, 0.f, 1.f );
    vtx->setQuad(1,0,0,    1.f, 0.f, 0.f, 1.f );
    vtx->setQuad(2,0,0,    0.f, 1.f, 0.f, 1.f );
  
    vtx_spec = new NPYBufferSpec(vtx->getBufferSpec()); 

    assert( vtx_spec->bufferByteLength == 3*4*4 + 5*16 ) ; 
    assert( vtx_spec->headerByteLength == 5*16 ) ; 


    glm::uvec4 iidx(0,1,2,0) ; 
    idx = NPY<unsigned>::make(4) ; 
    idx->zero();
    idx->setQuadU( iidx, 0,0,0 );
 
    idx_spec = new NPYBufferSpec(idx->getBufferSpec()); 

    assert( idx_spec->bufferByteLength == 4*4 + 5*16 ) ; 
    assert( idx_spec->headerByteLength == 5*16 ) ; 
}



