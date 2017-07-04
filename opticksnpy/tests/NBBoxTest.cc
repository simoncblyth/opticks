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
#include "NCylinder.hpp"
#include "NCone.hpp"

#include "NBBox.hpp"
#include "NScan.hpp"

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

     nbbox tbb = bb.make_transformed(tr) ; 

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

    nbbox tbb = bb.make_transformed(tr);
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


    unsigned verbosity = 2 ; 
    glm::vec3 origin(    0,0,0 );
    glm::vec3 direction( 1,0,0 );
    glm::vec3 range(     0,10,1 );

    nbbox bb0 ; 
    std::vector<float> sd0 ; 
    {
        ndifference ab = make_difference( &a, &b ); 
        ndifference abc = make_difference( &ab, &c ); 
        ndifference abcd = make_difference( &abc, &d ); 

        NScan scan(abcd, verbosity);
        scan.scan(sd0, origin, direction, range );

        abcd.get_composite_bbox(bb0);
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

        NScan scan(abcd, verbosity);
        scan.scan(sd1, origin, direction, range );

        abcd.get_composite_bbox(bb1);

        b.complement = false ;  
        c.complement = false ;  
        d.complement = false ;  
    }
    std::cout << "bb1 " << bb1.description() << std::endl ; 


    float epsilon = 1e-5f ; 
    float mxdf = SVec<float>::MaxDiff(sd0, sd1, false) ; 
    assert( std::fabs(mxdf) < epsilon ) ; 

}


void test_difference_bbox()
{
    LOG(info) << "test_difference_bbox " ;
 
    float radius = 10.f ; 
    float z1 = -1.f ; 
    float z2 = 1.f ; 

    ncylinder a = make_cylinder( radius     , z1, z2);   a.label = "A" ;     
    ncylinder b = make_cylinder( radius-1.f , z1, z2);   b.label = "B" ;   

    a.pdump("a");
    b.pdump("b");

    ndifference ab = make_difference( &a, &b ); 
    ab.dump("a-b");
}

void test_intersection_cone_difference_bbox()
{
    LOG(info) << "test_intersection_cone_difference_bbox " ;


    float radius = 10.f ; 
    float z1 = -1.f ; 
    float z2 = 1.f ; 
    ncylinder a = make_cylinder( radius     , z1, z2);   a.label = "A" ;     
    ncylinder b = make_cylinder( radius-1.f , z1, z2);   b.label = "B" ;   
    // wide flat ring 

    //a.pdump("a");
    //b.pdump("b");

    ndifference ab = make_difference( &a, &b ); 
    //ab.dump("a-b");

 
    // with cone in middle : too small to intersect 
    //float r1 = 5. ; 
    //float r2 = 4. ; 

    // jumbo cone that contains the cylinders
    //float r1 = 50. ; 
    //float r2 = 40. ; 

    // cone that intersects the cylinders
    float r1 = radius - 2.f ; 
    float r2 = radius + 2.f ; 


    ncone c = make_cone( r1, z1, r2, z2 );

    nintersection cab = make_intersection( &c, &ab );
    cab.dump("c*(a-b)");
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


void test_sdf()
{
    LOG(info) << "test_sdf" ; 

    glm::vec3 origin(0,0,0);
    glm::vec3 range(-20,21,5);

    nbbox bb0 = make_bbox( -10,-10,-10, 10,10,10 );
    bb0.scan_sdf(origin, range );

    nbbox bb1 = make_bbox( -5,-5,-5, 10,10,10 );
    bb1.scan_sdf(origin, range );

    nbbox bb2 = make_bbox( -5,-5,-5,  5,5,5 );
    bb2.scan_sdf(origin, range );

}
void test_sdf_transformed()
{
    LOG(info) << "test_sdf_transformed" ; 

    glm::vec3 origin(100,0,0);  // scanline around offset origin
    glm::vec3 range(-20,21,5);

    glm::vec3 tlate(100,0,0);
    const nmat4triple* t = nmat4triple::make_translate( tlate );

    // although scanline positions are offset, 
    // still get same as local results 
    // because the inverse transform is applied 
    // to the query point within the sdf evaluation

    nbbox bb0 = make_bbox( -10,-10,-10, 10,10,10 );
    bb0.scan_sdf(origin, range, t );

    nbbox bb1 = make_bbox( -5,-5,-5, 10,10,10 );
    bb1.scan_sdf(origin, range, t );

    nbbox bb2 = make_bbox( -5,-5,-5,  5,5,5 );
    bb2.scan_sdf(origin, range, t );
}


void test_from_points()
{
    LOG(info) << "test_from_points" ; 

    std::vector<glm::vec3> pts ; 

    pts.push_back( {1,0,0 } );
    pts.push_back( {2,0,0 } );

    nbbox bb = nbbox::from_points(pts);



    std::cout << bb.desc() << std::endl ;     

}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_bbox_transform();

    //const char* path = "$TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/transforms.npy" ;
    //test_bbox_transform_loaded( argc > 1 ? argv[1] : path );

    //test_overlap();
    //test_positive_form();

    //test_default_copy_ctor();


    //test_sdf();
    //test_sdf_transformed();
    //test_from_points();
    //test_difference_bbox();
    test_intersection_cone_difference_bbox() ;

    return 0 ; 
}



