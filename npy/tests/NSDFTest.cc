

#include "GLMFormat.hpp"
#include "NSDF.hpp"
#include "NCSG.hpp"
#include "NBox.hpp"
#include "NSphere.hpp"
#include "NGLMExt.hpp"
#include "N.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


// This is transforming model surface points with transform->t 
// only to immediately remove that with inverse transform-> within NSDF.
//
// Seems crazy in context of a single node.. but the reason
// for doing this is to allow comparison 
// between nodes which have different transforms.
//
// BUT how to structure that ?
//


void test_box_inscribed_in_sphere(const nmat4triple* tr, bool asrt)
{
    // testing the pbox.local points wrt to 
    // 
    //   box
    //       should all be very close to zero  
    //
    //  sphere 
    //       should mostly be -ve with some surface zeros at inscribe points
    //
    //
    //  notice that an inverse transform is applied to points prior to 
    //  evaluating the SDF so the values even with big scale transforms
    //  are all very similar as they are local frame ones
    //


    LOG(info) << "test_box_inscribed_in_sphere" ; 
    std::cout << gpresent("tr->t", tr->t) ;

    float r = 100.f ; 
    nsphere sph_ = make_sphere(0,0,0,r);
    sph_.label = "sph" ; 

    float h = sqrt(r*r/3.0) ;      // inscribe a box inside a sphere :  3* x^2  = r^2
    nbox box_ = make_box3(2*h,2*h,2*h); 
    box_.label = "box" ; 


    nbox fox_ = make_box3(h,h,h); // half sized box to be relatively placed on "floor" of other one 
    const nmat4triple* floor = nmat4triple::make_translate(0,0,-h/2.0 ) ;
    const nmat4triple* tr_floor = nmat4triple::product( tr, floor, false ); 
    fox_.label = "fox" ; 


    float epsilon = 1e-4 ; 

    N box(&box_, tr );    // placed box
    glm::uvec4 bbl = box.classify( box.local, epsilon, POINT_SURFACE );

    //if(bbl.w > 0) LOG(fatal) <<  "bbl.w fail " ; 
    if(asrt) assert( bbl.w == 0 );  // all points classified as POINT_SURFACE


    N sph(&sph_, tr  );    // placed sphere  
    glm::uvec4 sbl = sph.classify( box.local, epsilon, POINT_INSIDE|POINT_SURFACE );
    //if(sbl.w > 0)  LOG(fatal) <<  "sbl.w fail " ; 
    if(asrt) assert( sbl.w == 0 );  // all points as expected


    N fox(&fox_, tr_floor );    // box placed on floor of other one
    glm::uvec4 fbl = fox.classify( fox.local, epsilon, POINT_SURFACE );

    //if(fbl.w > 0) LOG(fatal) <<  "fbl.w fail " ; 
    if(asrt) assert( fbl.w == 0 );  // all points classified as POINT_SURFACE


    // classify points of the bigger box against SDF of the smaller
    // some on surface : from coincident bottom face, but mostly outside
    glm::uvec4 fbl2 = fox.classify( box.local, epsilon, POINT_SURFACE | POINT_OUTSIDE  );

    //if(fbl2.w > 0) LOG(fatal) <<  "fbl2.w fail " ; 
    if(asrt) assert( fbl2.w == 0 );


}



void test_concentric_spheres(const nmat4triple* tr, bool asrt)
{
    LOG(info) << "test_concentric_spheres" ;
    std::cout << gpresent("tr->t", tr->t) ;

    float ra = 100.f ; 
    nsphere a_ = make_sphere(0,0,0,ra);
    a_.label = "a_sph" ; 

    float rb = 99.f ; 
    nsphere b_ = make_sphere(0,0,0,rb);
    b_.label = "b_sph" ; 


    
    assert(ra > rb);
    float rc = (ra - rb)/2.f ; 
    nsphere c_ = make_sphere(0,0,0,rc);
    c_.label = "c_sph" ; 

    const nmat4triple* between = nmat4triple::make_translate(0,0,rb+rc ) ;
    const nmat4triple* tr_between = nmat4triple::product( tr, between, false ); 

    N a(&a_, tr  );    
    N b(&b_, tr  );    
    N c(&c_, tr_between );     // v.small sphere lodged between the concentric ones


    float epsilon = 1e-4 ; 


    // classify points of smaller b against a, expect all inside at SDF value ~ -1
    a.classify( b.local, epsilon, POINT_INSIDE );

    float ab_expect = rb - ra ; 
    if(asrt) 
    {
        assert(fabsf(a.nsdf.range[0] - ab_expect) < epsilon );
        assert(fabsf(a.nsdf.range[1] - ab_expect) < epsilon );
        assert(a.nsdf.tot.w == 0);
    }


    // classify points of larger a against b, expect all outside at SDF value ~ 1
    b.classify( a.local, epsilon, POINT_OUTSIDE );

    float ba_expect = ra - rb ; 
    if(asrt) 
    {
        assert(fabsf(b.nsdf.range[0] - ba_expect) < epsilon );
        assert(fabsf(b.nsdf.range[1] - ba_expect) < epsilon );
        assert(b.nsdf.tot.w == 0);
    }



    // c.local points are placed according to the c tr_between transform
    // which are then classified against the SDFs of a and b which 
    // use their inverse transforms 

    a.classify( c.local, epsilon, POINT_INSIDE|POINT_SURFACE ); 
    b.classify( c.local, epsilon, POINT_OUTSIDE|POINT_SURFACE ); 

    c.dump_points("c.dump_points");

}



void make_transforms( std::vector<const nmat4triple*>& trs )
{
    const nmat4triple* id = nmat4triple::make_identity() ;
    const nmat4triple* tr = nmat4triple::make_translate(0,0,50) ;
    const nmat4triple* ro = nmat4triple::make_rotate(1,1,1,45) ;
    const nmat4triple* sc = nmat4triple::make_scale(10,20,30) ;
   // scooting geometry off by 1,000,000 length units messes up the comparisons
    const nmat4triple* tr4 = nmat4triple::make_translate(0,0,1e4) ;
    const nmat4triple* tr5 = nmat4triple::make_translate(0,0,1e5) ;
    const nmat4triple* tr6 = nmat4triple::make_translate(0,0,1e6) ;
    const nmat4triple* tr7 = nmat4triple::make_translate(0,0,1e7) ;

    trs.push_back(id);
    trs.push_back(tr);
    trs.push_back(ro);
    trs.push_back(sc);
    trs.push_back(tr4);
    trs.push_back(tr5);
    trs.push_back(tr6);
    trs.push_back(tr7);
}



void test_box_inscribed_in_sphere()
{
    std::vector<const nmat4triple*> trs ;
    make_transforms(trs);
    for(unsigned i=0 ; i < trs.size() ; i++)
        test_box_inscribed_in_sphere( trs[i],  i < 4 );
}


void test_concentric_spheres()
{
    std::vector<const nmat4triple*> trs ;
    make_transforms(trs);
    //unsigned n = trs.size() ;
    unsigned n = 1; 

    for(unsigned i=0 ; i < n ; i++)
        test_concentric_spheres( trs[i],  i < 4 );
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //test_box_inscribed_in_sphere();
    test_concentric_spheres(); 

    return 0 ; 
}






 
