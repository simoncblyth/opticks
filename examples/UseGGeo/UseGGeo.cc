#include "GMatrix.hh"
#include "GVector.hh"
#include "GBBox.hh"

#include "OPTICKS_LOG.hh"

void test_bbox_matrix_scaling()
{
    LOG(info) << "[" ;

    // hmm this way is too complicated...
    gfloat3 min = gfloat3(10.f, 10.f, 10.f );
    gfloat3 max = gfloat3(20.f, 20.f, 20.f );
    gbbox bb(min, max); 

    bb.Summary("bef");

    gfloat3 cen = bb.center();

    cen.Summary("cen");

    float factor = 1.1f ; 

    // homogenous translate then scale matrix (ie translation not scaled)
    GMatrix<float> m1( -cen.x, -cen.y, -cen.z, 1.f );
    GMatrix<float> m2(    0.f,  0.f,  0.f,    factor );
    GMatrix<float> m3(  cen.x,  cen.y,  cen.z, 1.f );

    GMatrix<float> m ;
    m *= m1 ;
    m *= m2 ;
    m *= m3 ;
    
    m1.Summary("m1"); 
    m2.Summary("m2"); 
    m3.Summary("m3"); 

    m.Summary("m"); 

    bb *= m ; 
     
    bb.Summary("aft");
    LOG(info) << "]" ;
}

void test_bbox_enlarge()
{
    LOG(info) << "[" ;

    gfloat3 min = gfloat3(10.f, 10.f, 10.f );
    gfloat3 max = gfloat3(20.f, 20.f, 20.f );
    gbbox bb(min, max); 
    bb.Summary("bef");
    bb.enlarge(1.0f) ;  // factor of the extent
    bb.Summary("aft");

    LOG(info) << "]" ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    test_bbox_matrix_scaling();
    test_bbox_enlarge();
    return 0 ; 
}


