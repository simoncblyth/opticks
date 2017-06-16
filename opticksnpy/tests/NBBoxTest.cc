/*

NBBoxTest /tmp/blyth/opticks/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/transforms.npy 
*/


#include <cmath>
#include "SVec.hh"

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


void test_overlap()
{
    LOG(info) << "test_overlap" ; 

    {
        std::cout << std::endl << "totally not overlapping... " << std::endl ; 

        nbox _a = make_box(0.f,0.f,0.f,1.f);      
        _a.pdump("_a");

        nbbox a = _a.bbox();
        a.dump("a");

        nbox _b = make_box(3.f,3.f,3.f,1.f);      
        _b.pdump("_b");

        nbbox b = _b.bbox();
        b.dump("b");

        nbbox ab ;
        assert( a.has_overlap(b) == false );
        assert( a.find_overlap(ab, b) == false );
    }


    {
        std::cout << std::endl << "single point of overlap" << std::endl ; 

        nbox _a = make_box(0.f,0.f,0.f,1.f);      
        _a.pdump("_a");

        nbbox a = _a.bbox();
        a.dump("a");

        nbox _b = make_box(2.f,2.f,2.f,1.f);      
        _b.pdump("_b");

        nbbox b = _b.bbox();
        b.dump("b");

        nbbox ab ;
        assert( a.has_overlap(b) == true );
        assert( a.find_overlap(ab, b) == true );

        ab.dump("ab");
    }



    {
        std::cout << std::endl << "b contained inside a" << std::endl ; 

        nbox _a = make_box(0.f,0.f,0.f,1.f);      
        _a.pdump("_a");

        nbbox a = _a.bbox();
        a.dump("a");

        nbox _b = make_box(0.5f,0.5f,0.5f,0.5f);      
        _b.pdump("_b");

        nbbox b = _b.bbox();
        b.dump("b");

        nbbox ab ;
        assert( a.has_overlap(b) == true );
        assert( a.find_overlap(ab, b) == true );

        ab.dump("ab");
        assert( ab == b );
    }
 


    {
        std::cout << std::endl << "substantial overlap" << std::endl ; 

        nbox _a = make_box(0.f,0.f,0.f,1.f);      
        _a.pdump("_a");

        nbbox a = _a.bbox();
        a.dump("a");

        nbox _b = make_box(0.5f,0.5f,0.5f,1.f);      
        _b.pdump("_b");

        nbbox b = _b.bbox();
        b.dump("b");

        nbbox ab ;
        assert( a.has_overlap(b) == true );
        assert( a.find_overlap(ab, b) == true );

        ab.dump("ab");
        assert( ab.min == b.min );
        assert( ab.max == a.max );
    }
}



void test_positive_form()
{
/*
          -             *
         / \           / \
        -   d         *   !d
       / \           / \
      -   c         *   !c 
     / \           / \
    a   b         a   !b   

*/

    LOG(info) << "test_positive_form" ; 

    nbox a = make_box(0.f,0.f,0.f,10.f);   a.label = "A" ;     
    nbox b = make_box(5.f,0.f,0.f,0.01f);    b.label = "B" ;   
    nbox c = make_box(0.f,5.f,0.f,0.01f);    c.label = "C" ;   
    nbox d = make_box(0.f,0.f,5.f,0.01f);    d.label = "D" ;     

    a.pdump("a");
    b.pdump("b");
    c.pdump("c");
    d.pdump("d");

    nbbox a_bb = a.bbox();
    nbbox b_bb = b.bbox();
    nbbox c_bb = c.bbox();
    nbbox d_bb = d.bbox();


    std::cout << a.desc() << " " << a_bb.desc() << std::endl ; 
    std::cout << b.desc() << " " << b_bb.desc() << std::endl ; 
    std::cout << c.desc() << " " << c_bb.desc() << std::endl ; 
    std::cout << d.desc() << " " << d_bb.desc() << std::endl ; 


    bool dump = true ; 
    glm::vec3 origin(    0,0,0 );
    glm::vec3 direction( 1,0,0 );
    glm::vec3 range(     0,10,1 );

    nbbox bb0 ; 
    std::vector<float> sd0 ; 
    {
        ndifference ab = make_difference( &a, &b ); 
        ndifference abc = make_difference( &ab, &c ); 
        ndifference abcd = make_difference( &abc, &d ); 

        nnode::Scan(sd0,  abcd, origin, direction, range, dump );
        abcd.composite_bbox(bb0);
    }
    std::cout << "bb0 " << bb0.description() << std::endl ; 


    nbbox bb1 ; 
    std::vector<float> sd1 ; 
    {
        b.complement = true ;  
        c.complement = true ;  
        d.complement = true ;  

        nintersection ab = make_intersection( &a, &b ); 
        nintersection abc = make_intersection( &ab, &c ); 
        nintersection abcd = make_intersection( &abc, &d ); 

        nnode::Scan(sd1,  abcd, origin, direction, range, dump );
        abcd.composite_bbox(bb1);

        b.complement = false ;  
        c.complement = false ;  
        d.complement = false ;  
    }
    std::cout << "bb1 " << bb1.description() << std::endl ; 


    float epsilon = 1e-5f ; 
    float mxdf = SVec<float>::MaxDiff(sd0, sd1, false) ; 
    assert( std::fabs(mxdf) < epsilon ) ; 

}


void test_default_copy_ctor()
{
    LOG(info) << "test_default_copy_ctor" ; 

    nbbox bb ; 

    bb.min = {-10,-10,-10} ;
    bb.max = { 10, 10, 10} ;
    bb.side = bb.max - bb.min ; 

    bb.invert = true ; 
    bb.empty = false ; 


    nbbox cbb(bb) ;

    std::cout << " bb  " << bb.desc() << std::endl ; 
    std::cout << " cbb " << cbb.desc() << std::endl ; 

}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_bbox_transform();

    //const char* path = "$TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/transforms.npy" ;
    //test_bbox_transform_loaded( argc > 1 ? argv[1] : path );

    //test_overlap();
    test_positive_form();

    test_default_copy_ctor();


    return 0 ; 
}



