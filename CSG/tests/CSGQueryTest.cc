#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::LoadGeom(); 

    CSGQuery* q = new CSGQuery(fd); 

    CSGDraw* d = new CSGDraw(q) ;

    d->draw("CSGQueryTest");

    float3 ray_origin =   make_float3(  5.f, 5.f, 0.f ); 

    float3 ray_direction = make_float3( 1.f, 0.f, 0.f ); 

    float t_min = 0.f ; 

    unsigned gsid = 0 ; 

    quad4 isect ; 

    bool valid_intersect = q->intersect(isect, t_min, ray_origin, ray_direction , gsid ); 

    std::cout << CSGQuery::Desc( isect, "trial", &valid_intersect ) << std::endl ; 

    return 0 ; 
}

