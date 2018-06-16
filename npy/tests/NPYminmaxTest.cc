
#include <iostream>

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

NPY<float>* make_vtx()
{
    NPY<float>* vtx = NPY<float>::make(3, 4) ; 
    vtx->zero();
    vtx->setQuad(0,0,0,    0.f, 0.f, 0.f, 1.f );
    vtx->setQuad(1,0,0,    1.f, 0.f, 0.f, 1.f );
    vtx->setQuad(2,0,0,    0.f, 1.f, 0.f, 1.f );

    return vtx ; 
}

NPY<unsigned>* make_idx()
{
    NPY<unsigned>* idx = NPY<unsigned>::make(3, 1) ;
    idx->zero();

    //            i  j  k  l   value
    idx->setValue(0, 0, 0, 0,  0);
    idx->setValue(1, 0, 0, 0,  1);
    idx->setValue(2, 0, 0, 0,  2);

    return idx ;
} 

void test_minmax_vector_i()
{
    NPY<unsigned>* idx = make_idx();
    unsigned nelem = idx->getNumElements() ; 
    assert( nelem == 1 );  

    std::vector<unsigned> imin ; 
    std::vector<unsigned> imax ; 
    idx->minmax(imin, imax); 

    assert( imin.size() == 1 );
    assert( imax.size() == 1 );

    assert( imin[0] == 0 );
    assert( imax[0] == 2 );
}




void test_minmax_vector()
{
    NPY<float>* vtx = make_vtx();
    unsigned nelem = vtx->getNumElements() ; 
    assert( nelem == 4 );  

    std::vector<float> vmin ; 
    std::vector<float> vmax ; 
    vtx->minmax(vmin, vmax); 

    assert( vmin[0] == 0.f );
    assert( vmin[1] == 0.f );
    assert( vmin[2] == 0.f );
    assert( vmin[3] == 1.f );

    assert( vmax[0] == 1.f );
    assert( vmax[1] == 1.f );
    assert( vmax[2] == 0.f );
    assert( vmax[3] == 1.f );

    LOG(info) << "." ; 
    std::cout << " vmin : " ; 
    std::copy( vmin.begin(), vmin.end(), std::ostream_iterator<float>(std::cout, " ")) ;
    std::cout << std::endl ; 

    std::cout << " vmax : " ; 
    std::copy( vmax.begin(), vmax.end(), std::ostream_iterator<float>(std::cout, " ")) ;
    std::cout << std::endl ; 
}




void test_minmax_ntvec()
{
    NPY<float>* vtx = make_vtx();

    ntvec4<float> vtx_min ;   
    ntvec4<float> vtx_max ;   
    vtx->minmax4( vtx_min, vtx_max );

    LOG(info) << "vtx_min " << vtx_min.desc() ; 
    LOG(info) << "vtx_max " << vtx_max.desc() ; 

    assert( vtx_min.x == 0.f );  
    assert( vtx_min.y == 0.f );  
    assert( vtx_min.z == 0.f );  
    assert( vtx_min.w == 1.f );

    assert( vtx_max.x == 1.f );  
    assert( vtx_max.y == 1.f );  
    assert( vtx_max.z == 0.f );  
    assert( vtx_max.w == 1.f );
}



int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv); 

    test_minmax_ntvec();
    test_minmax_vector();
    test_minmax_vector_i();

    return 0 ; 
}

