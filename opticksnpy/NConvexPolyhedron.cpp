
#include <cfloat>
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "NConvexPolyhedron.hpp"
#include "NBBox.hpp"


float nconvexpolyhedron::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;

    unsigned num_planes = planes.size(); 
    float dmax = -FLT_MAX ; 

   // * does this need to distinguish cases of being inside and outside the planes
   // * is this assuming origin is inside the convexpolyhedron ? 

    int verbosity = 0 ; 

    for(unsigned i=0 ; i < num_planes ; i++)
    {
        const nvec4& plane = planes[i]; 
        glm::vec3 pnorm(plane.x, plane.y, plane.z );
        assert( plane.w > 0.f );// <-- TODO: assert elsewhere in lifecycle
        float pdist = plane.w ; 

        float d0 = glm::dot(pnorm, glm::vec3(q)) ;   // distance from q to the normal plane thru origin
        if(d0 == 0.f) continue ; 
         
        float d = d0 - pdist ; 

        if(verbosity > 2)
        std::cout 
             << " pl: " << plane.desc()
             << " " << std::setw(10) << d 
             << std::endl 
             ; 

        if(d > dmax) dmax = d ;      
    }

    if(verbosity > 2)
    std::cout << std::endl ; 

    return dmax ; 
} 


nbbox nconvexpolyhedron::bbox() const 
{
    nbbox bb = make_bbox();
    bb.min = make_nvec3(param2.f.x, param2.f.y, param2.f.z) ;
    bb.max = make_nvec3(param3.f.x, param3.f.y, param3.f.z) ;
    bb.side = bb.max - bb.min ; 
    return gtransform ? bb.transform(gtransform->t) : bb ; 
}



bool nconvexpolyhedron::intersect( const float t_min, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect ) const 
{
    float t0 = -FLT_MAX ; 
    float t1 =  FLT_MAX ; 

    glm::vec3 t0_normal(0.f);
    glm::vec3 t1_normal(0.f);

    int verbosity = 0 ; 
    unsigned num_planes = planes.size(); 

    for(unsigned i=0 ; i < num_planes && t0 < t1  ; i++)
    {
        const nvec4& plane = planes[i]; 
        glm::vec3 pnorm(plane.x, plane.y, plane.z );
        assert( plane.w > 0.f );// <-- TODO: assert elsewhere in lifecycle
        float pdist = plane.w ;

        float denom = glm::dot(pnorm, ray_direction);
        if(denom == 0.f) continue ;   

        float t_cand = (pdist - glm::dot(pnorm, ray_origin))/denom;
   
        if(verbosity > 2) std::cout << "nconvexpolyhedron::intersect"
                                    << " i " << i
                                    << " t_cand " << t_cand 
                                    << " denom " << denom 
                                    << std::endl ; 


        if( denom < 0.f)  // ray opposite to normal, ie ray from outside entering
        {
            if(t_cand > t0)
            {
                t0 = t_cand ;
                t0_normal = pnorm ;
            }
        } 
        else 
        {          // ray same hemi as normal, ie ray from inside exiting 
            if(t_cand < t1)
            {
                t1 = t_cand ;
                t1_normal = pnorm ;
            }
        }
    }

    // should always have both a t0 (ingoing-intersect) and a t1 (outgoing-intersect)
    // as no matter where from the ray must intersect two planes...
    //
    // similar the slab method, when the t ordering is wrong that means plane intersects
    // outside of the solid

    bool valid_intersect = t0 < t1 ; 
    if(valid_intersect)
    {
        if( t0 > t_min )
        {
            isect.x = t0_normal.x ; 
            isect.y = t0_normal.y ; 
            isect.z = t0_normal.z ; 
            isect.w = t0 ; 
        } 
        else if( t1 > t_min )
        {
            isect.x = t1_normal.x ; 
            isect.y = t1_normal.y ; 
            isect.z = t1_normal.z ; 
            isect.w = t1 ; 
        }
   }
   return valid_intersect ;
}





glm::vec3 nconvexpolyhedron::gseedcenter()
{
    glm::vec3 center(0.f,0.f,0.f) ;
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

glm::vec3 nconvexpolyhedron::gseeddir()
{
    glm::vec4 dir(1.,0.,0.,0.); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


void nconvexpolyhedron::pdump(const char* msg, int verbosity )
{
    unsigned num_planes = planes.size();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " gtransform? " << !!gtransform
              << " num_planes " << num_planes
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;



}




