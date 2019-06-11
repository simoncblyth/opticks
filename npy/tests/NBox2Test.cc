// TEST=NBox2Test om-t


#include <cstdlib>
#include <cfloat>

#include "NBox.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"

void test_adjustToFit()
{
    float h = 10.f ; 
    nbox* box = make_box3(2*h,2*h,2*h); 

    box->verbosity = 3 ;  
    box->pdump("original make_box3(2*h,2*h,2*h)");

    nbbox bb = box->bbox_model();

    LOG(info) << " bb (natural bbox) " << bb.desc() ; 

    nbbox cc = make_bbox( bb.min.x - 1 , bb.min.y - 1, bb.min.z - 1, bb.max.x + 1 , bb.max.y + 1, bb.max.z + 1 ) ;

    LOG(info) << " cc ( enlarged bbox ) " << cc.desc() ; 

    box->adjustToFit( cc , 1.f, 0.f );    

    box->pdump("after adjustToFit make_box3(2*h,2*h,2*h)");

    nbbox cc2 = box->bbox_model();

    assert( cc.is_equal(cc2) );

    box->adjustToFit( cc , 1.f, 1.f );    
   
    nbbox cc3 = box->bbox_model();

    assert( !cc.is_equal(cc3) );
 
    LOG(info) << " cc3 ( enlarged bbox ) " << cc.desc() ; 
}



void test_adjustToFit_box()
{
    nbox* box = make_box( 0.f, 0.f, -5.f, 10.f ); 
    box->verbosity = 3 ;  
    box->pdump("make_box(0,0,-5,10)");


    float sz = 100.f ; 
    float dz = -50.f ; 
    nbbox bb = make_bbox( -sz, -sz, -sz + dz,   sz, sz, sz + dz ); 
    nbox* xbox = make_box( 0.f, 0.f, dz, sz ); 

    LOG(info) << " bb " << bb.description() ; 

    float scale = 1.f ; 
    float delta = 0.f ; 
    box->adjustToFit( bb, scale , delta );  
    // ignores initial box, simply changes it to correspond to the bb 
    // BUT shifts of the bbox are honoured

    box->pdump("after adjustToFit "); 

    assert( box->is_equal(*xbox) ); 
}

void test_adjustToFit_box3()
{
    nbox* box = make_box3( 10.f, 10.f, 20.f ); 
    box->verbosity = 3 ;  
    box->pdump("make_box3(10,10,20)");


    float sz = 100.f ; 
    float dz = -50.f ; 
    nbbox bb = make_bbox( -sz, -sz, -sz + dz,   sz, sz, sz + dz ); 
    // this bb is shifted down in z 


    nbox* ybox = make_box3(2.f*sz, 2.f*sz, 2.f*sz ); 

    LOG(info) << " bb " << bb.description() ; 

    float scale = 1.f ; 
    float delta = 0.f ; 
    box->adjustToFit( bb, scale , delta );  

    // ignores initial box, simply changes it to correspond to the bb 
    // also ignores any shifts in the bbox

    box->pdump("after adjustToFit "); 

    assert( box->is_equal(*ybox) ); 
}

void test_box()
{
    float sz = 10.f ; 
    float dz = -sz/2.f ;   // push down in z

    nbox* ybox = make_box(0.f, 0.f, dz, sz ); 
    ybox->pdump("make_box(0,0,-5, 10)");
    nbbox bb = ybox->bbox_model();
    LOG(info) << " bb " << bb.description() ; 

    nbbox xbb = make_bbox( -sz, -sz, -sz+dz  ,   sz, sz, sz+dz  ); 

    assert( bb.is_equal(xbb) );
}

void test_box3()   // always symmetric
{
    float sz = 10.f ; 

    nbox* box = make_box3( sz, sz, 2.f*sz ); 
    box->pdump("make_box3(10,10,20)");
    nbbox bb = box->bbox_model();
    LOG(info) << " bb " << bb.description() ; 

    nbbox xbb = make_bbox( -sz/2.f, -sz/2.f, -sz  ,   sz/2.f, sz/2.f, sz  ); 
    assert( bb.is_equal(xbb) );
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_adjustToFit();
    //test_adjustToFit_box();
    test_adjustToFit_box3();

    //test_box();
    //test_box3();

    return 0 ; 
}



