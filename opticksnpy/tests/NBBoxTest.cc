/*

NBBoxTest /tmp/blyth/opticks/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/transforms.npy 
*/


#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY.hpp"
#include "NGenerator.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


glm::mat4 make_test_matrix()
{
    glm::vec3 axis(0,0,1);
    glm::vec3 tlate(0,0,100.);
    float angle = 45.f ; 

    glm::mat4 tr(1.0f) ; 
    tr = glm::rotate(tr, angle, axis );
    tr = glm::translate(tr, tlate );

    std::cout << gpresent("tr", tr) << std::endl ; 
    return tr ; 
}

void test_bbox_transform()
{
     glm::mat4 tr = make_test_matrix();

     nbox a = make_box(0.f,0.f,0.f,100.f);      
     nbbox bb = a.bbox();

     nbbox tbb0 ;
     nbbox::transform_brute( tbb0, bb, tr );

     nbbox tbb1 ;
     nbbox::transform( tbb1, bb, tr );

     nbbox tbb = bb.transform(tr) ; 

     assert( tbb == tbb0 );
     assert( tbb == tbb1 );

     std::cout << " tbb  " << tbb.desc() << std::endl ; 
     std::cout << " tbb0 " << tbb0.desc() << std::endl ; 
     std::cout << " tbb1 " << tbb1.desc() << std::endl ; 
}


void test_bbox_transform_loaded(const char* path)
{
    NPY<float>* buf = NPY<float>::load(path);
    if(!buf) return ; 
    buf->dump();

    glm::mat4 tr = buf->getMat4(0);
    std::cout << gpresent("tr",tr ) << std::endl ; 

    std::cout << gpresent("tr[0]",tr[0] ) << std::endl ; 
    std::cout << gpresent("tr[1]",tr[1] ) << std::endl ; 
    std::cout << gpresent("tr[2]",tr[2] ) << std::endl ; 
    std::cout << gpresent("tr[3]",tr[3] ) << std::endl ; 

    nbox a = make_box(0.f,0.f,0.f,100.f);      
    nbbox bb = a.bbox();
    std::cout << "bb " <<  bb.desc() << std::endl ; 

    nbbox tbb = bb.transform(tr);
    std::cout << "tbb " <<  tbb.desc() << std::endl ; 

}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_bbox_transform();

    const char* path = "$TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/transforms.npy" ;
    test_bbox_transform_loaded( argc > 1 ? argv[1] : path );

    return 0 ; 
}



