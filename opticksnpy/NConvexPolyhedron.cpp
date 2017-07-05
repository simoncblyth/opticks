
#include <cfloat>

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "NConvexPolyhedron.hpp"
#include "Nuv.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"



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
        const glm::vec4& plane = planes[i]; 
        glm::vec3 pnorm(plane.x, plane.y, plane.z );
        assert( plane.w > 0.f );// <-- TODO: assert elsewhere in lifecycle
        float pdist = plane.w ; 

        float d0 = glm::dot(pnorm, glm::vec3(q)) ;   // distance from q to the normal plane thru origin
        if(d0 == 0.f) continue ; 
         
        float d = d0 - pdist ; 

        if(verbosity > 2)
        std::cout 
             << " pl: " << gpresent(plane)
             << " " << std::setw(10) << d 
             << std::endl 
             ; 

        if(d > dmax) dmax = d ;      
    }

    if(verbosity > 2)
    std::cout << std::endl ; 

    float sd = dmax ;

    return complement ? -sd : sd ; 
} 


nbbox nconvexpolyhedron::bbox() const 
{
    glm::vec3 mi(param2.f.x, param2.f.y, param2.f.z) ;
    glm::vec3 mx(param3.f.x, param3.f.y, param3.f.z) ;
    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
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
        const glm::vec4& plane = planes[i]; 
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


void nconvexpolyhedron::pdump(const char* msg) const 
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




unsigned nconvexpolyhedron::par_nsurf() const 
{
   return planes.size() ; 
}
int nconvexpolyhedron::par_euler() const 
{
   return 2 ; 
}
unsigned nconvexpolyhedron::par_nvertices(unsigned /*nu*/, unsigned /*nv*/) const 
{
   return planes.size();
}

glm::vec3 nconvexpolyhedron::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s < planes.size() );

    glm::vec4 pl = planes[s];

    glm::vec3 norm(pl.x,pl.y,pl.z) ; 
    float dist = pl.w ; 

    float epsilon = 1e-5 ; 
    assert( fabsf(glm::length(norm) - 1.f ) < epsilon ); 

    glm::vec3 pos = norm*dist ; 
  
    return pos ; 
}

void nconvexpolyhedron::set_bbox(const nbbox& bb)
{
    param2.f.x = bb.min.x ;
    param2.f.y = bb.min.y ;
    param2.f.z = bb.min.z ;

    param3.f.x = bb.max.x ;
    param3.f.y = bb.max.y ;
    param3.f.z = bb.max.z ;
}




nconvexpolyhedron* nconvexpolyhedron::make_trapezoid(float z, float x1, float y1, float x2, float y2 ) // static
{
   /*
    z-order verts


                  6----------7
                 /|         /|
                / |        / |
               4----------5  |
               |  |       |  |                       
               |  |       |  |         Z    
               |  2-------|--3         |  Y
               | /        | /          | /
               |/         |/           |/
               0----------1            +------ X
                         

    x1: x length at -z
    y1: y length at -z

    x2: x length at +z
    y2: y length at +z

    z:  z length

    */

    std::vector<glm::vec3> v(8) ; 
                                    // ZYX
    v[0] = { -x1/2., -y1/2. , -z } ;  // 000
    v[1] = {  x1/2., -y1/2. , -z } ;  // 001 
    v[2] = { -x1/2.,  y1/2. , -z } ;  // 010
    v[3] = {  x1/2.,  y1/2. , -z } ;  // 011

    v[4] = { -x2/2., -y2/2. ,  z } ;  // 100
    v[5] = {  x2/2., -y2/2. ,  z } ;  // 101
    v[6] = { -x2/2.,  y2/2. ,  z } ;  // 110
    v[7] = {  x2/2.,  y2/2. ,  z } ;  // 111

    std::vector<glm::vec4> p(6) ; 

    p[0] = make_plane( v[3], v[7], v[5] ) ; // +X  
    p[1] = make_plane( v[0], v[4], v[6] ) ; // -X
    p[2] = make_plane( v[2], v[6], v[7] ) ; // +Y
    p[3] = make_plane( v[1], v[5], v[4] ) ; // -Y
    p[4] = make_plane( v[5], v[7], v[6] ) ; // +Z
    p[5] = make_plane( v[3], v[1], v[0] ) ; // -Z

    nconvexpolyhedron* cpol = make_convexpolyhedron_ptr() ;
    std::copy( p.begin() , p.end() , std::back_inserter(cpol->planes) ) ;

    nbbox bb = nbbox::from_points( v );
    cpol->set_bbox(bb);

    return cpol ; 
}


nconvexpolyhedron* nconvexpolyhedron::make_transformed( const glm::mat4& t  ) const 
{
    nconvexpolyhedron* cpol = make_convexpolyhedron_ptr() ;

    for(unsigned i=0 ; i < planes.size() ; i++ )
    {
        nplane pl = make_plane(planes[i]);
        glm::vec4 tpl = pl.make_transformed(t);
        cpol->planes.push_back(tpl);  
    }

    nbbox bb = bbox();
    nbbox tbb = bb.make_transformed(t);
    cpol->set_bbox(tbb);

    return cpol ; 
}





