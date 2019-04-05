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




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_adjustToFit();

    return 0 ; 
}



