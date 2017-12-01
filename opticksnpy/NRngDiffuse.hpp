#pragma once

#include <string>
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

template <typename T> class NPY  ;

class BRng ; 

/**

NRngDiffuse
============

This is using a sampling approach 
similat to that used all over G4OpBoundaryProcess


Other potential approaches:

* http://www.rorydriscoll.com/2009/01/07/better-sampling/

**/


class NPY_API NRngDiffuse
{
    public:
         NRngDiffuse(unsigned seed, float ctmin, float ctmax);
         std::string desc() const ; 

         float diffuse( glm::vec4& v, int& trials, const glm::vec3& dir) ;
         void uniform_sphere(glm::vec4& u );


        NPY<float>* uniform_sphere_sample(unsigned n);
        NPY<float>* diffuse_sample(unsigned n, const glm::vec3& dir);

    private:
         const float m_pi  ; 
         unsigned    m_seed ; 
         BRng*       m_uazi ; 
         BRng*       m_upol ; 
         BRng*       m_unct ; 
    

};
