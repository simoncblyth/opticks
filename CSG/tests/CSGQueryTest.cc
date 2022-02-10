#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"
#include "NP.hh"

#include "OPTICKS_LOG.hh"

void XScan(const CSGQuery* q )
{
    float3 ray_origin, ray_direction_0, ray_direction_1  ; 
    float t_min ; 

    qvals(ray_origin   , "ORI", "3,0,0" );  
    qvals(ray_direction_0, "DIR0", "0,0,1" );  
    qvals(ray_direction_1, "DIR1", "0,0,-1" );  
    qvals(t_min        , "TMIN", "0" ); 

    ray_direction_0 = normalize(ray_direction_0); 
    ray_direction_1 = normalize(ray_direction_1); 

    unsigned gsid = 0 ; 
    int num = 10 ; 

    std::vector<quad4> isects(num*2) ;
 
    for(int i=0 ; i < num ; i++)
    {
         ray_origin.x = float(i - num/2) ;  

         quad4& isect_0 = isects[2*i+0] ; 
         quad4& isect_1 = isects[2*i+1] ; 

         bool valid_intersect_0 = q->intersect(isect_0, t_min, ray_origin, ray_direction_0 , gsid ); 
         std::cout << CSGQuery::Desc( isect_0, "trial_0", &valid_intersect_0 ) << std::endl ; 

         bool valid_intersect_1 = q->intersect(isect_1, t_min, ray_origin, ray_direction_1 , gsid ); 
         std::cout << CSGQuery::Desc( isect_1, "trial_1", &valid_intersect_1 ) << std::endl ; 
    }

    NP::Write("/tmp", "CSGQueryTest.npy",  (float*)isects.data(), isects.size(), 4, 4 ); 
}


void One( const CSGQuery* q )
{
    float3 ray_origin, ray_direction ; 
    float t_min ; 

    qvals(ray_origin    , "ORI", "3,0,0" );  
    qvals(ray_direction , "DIR", "1,0,0" );  
    qvals(t_min         , "TMIN", "0" ); 
    unsigned gsid = 0 ; 

    std::vector<quad4> isects(1) ;

    quad4& isect = isects[0] ; 

    bool valid_intersect = q->intersect(isect, t_min, ray_origin, ray_direction, gsid ); 
    std::cout << CSGQuery::Desc( isect, "trial_1", &valid_intersect ) << std::endl ; 

    NP::Write("/tmp", "CSGQueryTest.npy",  (float*)isects.data(), isects.size(), 4, 4 ); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char mode = argc > 1 ? argv[1][0] : 'O' ; 

    LOG(info) << " mode " << mode ; 

    CSGFoundry* fd = CSGFoundry::LoadGeom(); 

    CSGQuery* q = new CSGQuery(fd); 

    CSGDraw* d = new CSGDraw(q) ;

    d->draw("CSGQueryTest");

    switch(mode)
    {
        case 'O': One(q)   ; break ; 
        case 'X': XScan(q) ; break ; 
    }

    return 0 ; 
}

