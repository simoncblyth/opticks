#include <iostream>
#include <glm/glm.hpp>

#include "NImplicitMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"



struct sphere_functor 
{
   sphere_functor( float x, float y, float z, float r, bool negate)
       :   
       center(x,y,z),
       radius(r),
       negate(negate)
   {   
   }   

   float operator()( float x, float y, float z) const
   {   
       glm::vec3 p(x,y,z) ;
       float d = glm::distance( p, center );
       float v = d - radius ; 
       return negate ? -v : v  ;
   }   

   std::string desc();


   glm::vec3 center ; 
   float     radius ; 
   bool      negate ; 

};


NTrianglesNPY* test_sphere_node()
{
    nsphere* sph = new nsphere(make_sphere(0,0,0, 10)) ;
    nbbox bb = sph->bbox();

    LOG(info) << "test_sphere_node bb:" << bb.desc() ;

    int resolution = 100 ; 
    int verbosity = 1 ; 
    float bb_scale = 1.01 ; 

    NImplicitMesher im(sph, resolution, verbosity, bb_scale);
    NTrianglesNPY* tris = im();
    assert(tris);

    return tris ; 
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_sphere_node();
    //test_sphere_functor();

    return 0 ; 
}
