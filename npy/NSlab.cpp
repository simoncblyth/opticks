/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include <glm/gtx/component_wise.hpp> 
#include "Nuv.hpp"
#include "NSlab.hpp"

/*

Consider CSG difference of near and far halfspaces

        difference(near-half-space,far-half-space) = max( dnear, -dfar )



                         ^      
                         |
                  -------+---------   
                                  -ve: inside far halfspace
                                      
                         ^       
                         |         
                  -------+---------   
                                  -ve: inside near halfspace
                        
 
                         .  origin


  CSG difference "max(l,-r)" near and far:

           far-halfspace - near-halfspace  = slab 

  CSG intersect "max(l,r)" near and -far (complement of far)


  Not so easy to imagine CSG difference that leaves you with 
  negative space, so instead just flip the order and negate

     A - B =>  -(B - A)

   So you get everything not in the slab.

  View the planes as a half-spaces want to form CSG intersection


  CSG 
      union(l,r)     ->  min(l,r)
      intersect(l,r) ->  max(l,r)
      difference(l,r) -> max(l,-r)

*/

float nslab::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;

    glm::vec3 n = normal();
    float a_ = a();
    float b_ = b();

    float d = glm::dot(n, glm::vec3(q)) ; // distance from q to the nearest point on plane thru origin
    float da =  d - a_ ;             
    float db =  d - b_  ;   // b > a
  
    float sd = fmaxf(db, -da)   ;

    return complement ? -sd : sd ; 
} 


/*

Ray
     q = ray_origin + t * ray_direction 

Planes
     q.n = far  
     q.n = near     "-near" for outward slab normals? 
 
Intersects
      q.n = ray_origin.n + t * ray_direction.n 
      t = (q.n - ray_origin.n) / ray_direction.n

Just because the ray intersects the slab doesnt mean its valid, 
there are 3 possibilities depending in where t_min lies wrt t_near and t_far.

                         t_near       t_far   
                           |           |
                 -----1----|----2------|------3---------->
                           |           |

*/


bool nslab::intersect( const float t_min, const glm::vec3& ray_origin, const glm::vec3& ray_direction, glm::vec4& isect )
{

    glm::vec3 n = normal();
    float a_ = a();
    float b_ = b();

    float idn = 1.f/glm::dot(ray_direction, n );  // <-- infinite when ray is perpendicular to normal 
    float on = glm::dot(ray_origin, n );

    float ta = (a_ - on)*idn ; 
    float tb = (b_ - on)*idn ; 

    float t_near = fminf(ta,tb);  // order the intersects 
    float t_far  = fmaxf(ta,tb);

    float t_cand = t_near > t_min ?  t_near : ( t_far > t_min ? t_far : t_min ) ;

    bool valid_intersect = t_cand > t_min ; 
    bool b_hit = t_cand == tb ;    
     
     std::cout 
         << "nslab::intersect" 
         << " a " << a_ 
         << " b " << b_ 
         << " ta " << ta
         << " tb " << tb
         << " idn " << idn
         << " on " << on
         << " t_near " << t_near
         << " t_far " << t_far
         << " t_cand " << t_cand
         << " b_hit " << b_hit 
         << std::endl ; 

    if( valid_intersect )
    {
         isect.x = b_hit ? n.x : -n.x ; 
         isect.y = b_hit ? n.y : -n.y ; 
         isect.z = b_hit ? n.z : -n.z ; 
         isect.w = t_cand ;    
    }
    return valid_intersect  ; 
}


glm::vec3 nslab::center() const
{
    glm::vec3 n = normal();
    float a_ = a();
    float b_ = b();
    glm::vec3 center_ = n * (a_+b_)/2.f ;
    return center_ ; 
}


glm::vec3 nslab::gseedcenter()
{
    glm::vec3 center_ = center() ;
    return gtransform == NULL ? center_ : glm::vec3( gtransform->t * glm::vec4(center_, 1.f ) ) ;
}

glm::vec3 nslab::gseeddir()
{
    glm::vec3 n = normal();
    glm::vec4 dir(n,0); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}



void nslab::pdump(const char* msg) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " n " << normal()
              << " a  " << a()
              << " b (b>a) " << b()
              << " gtransform? " << !!gtransform
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}



unsigned nslab::par_nsurf() const 
{
   return 2 ; 
}
int nslab::par_euler() const 
{
   return 2 ; 
}
unsigned nslab::par_nvertices(unsigned /*nu*/, unsigned /*nv*/) const 
{
   return 0 ; 
}

glm::vec3 nslab::par_pos_model(const nuv& uv) const 
{
    glm::vec3 n = normal();

    float sz = 500.f ;  // arbitrary square patch region of the slab planes

    unsigned s = uv.s(); 
    float fu = sz*(uv.fu() - 0.5f) ;  
    float fv = sz*(uv.fv() - 0.5f) ;
 
    glm::vec3 offset = fu*udir + fv*vdir ;  

    glm::vec3 pos =  n * ( s == 0 ? a() : b() ) ; 
    pos += offset ; 

    return pos ; 
}


void nslab::define_uv_basis() 
{
    glm::vec3 n = normal();
    nglmext::_define_uv_basis(n, udir, vdir) ;
}


/*


                     | 
                     | 
           ----------b-----------
                     |
                     |
                     |
           ----------a-----------
                     |
                     |
                     |
                     0

*/



