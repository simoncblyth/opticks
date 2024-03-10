#pragma once
/**
SGLM_Arcball.h
================

* https://research.cs.wisc.edu/graphics/Courses/559-f2001/Examples/Gl3D/arcball-gems.pdf
* ~/opticks_refs/ken_shoemake_arcball_rotation_control_gem.pdf 

* http://www.talisman.org/~erlkonig/misc/shoemake92-arcball.pdf
* ~/opticks_refs/shoemake92-arcball.pdf

* :google:`ken shoemake arcball rotation control gem`

**/

#include <glm/glm.hpp>


struct SGLM_Arcball
{
    static glm::quat A2B_Arcball( const glm::vec3& a, const glm::vec3& b ) ; 
    static glm::quat A2B_Screen( const glm::vec2& a, const glm::vec2& b ) ; 
    static float ZProject(const glm::vec2& p ); 
}; 

/**
SGLM_Arcball::A2B_Arcball
-------------------------

* formula to form quat from two vecs direct from Ken Shoemake
  using properties of pure quaternions (which are points 
  on the equator of the hypersphere)

**/

inline glm::quat SGLM_Arcball::A2B_Arcball( const glm::vec3& a, const glm::vec3& b )
{
    glm::vec3 A = glm::normalize(a); 
    glm::vec3 B = glm::normalize(b); 
    return glm::quat( -glm::dot(A,B), glm::cross(A,B) ); 
}

inline glm::quat SGLM_Arcball::A2B_Screen( const glm::vec2& a, const glm::vec2& b )
{
    glm::vec3 A(a.x, a.y, ZProject(a) ); 
    glm::vec3 B(b.x, b.y, ZProject(b) ); 
    return A2B_Arcball(A, B) ; 
} 

/**
SGLM_Arcball::ZProject cf Trackball::project
-------------------------------------------------

What to do when outside unit circle is kinda arbitrary 
are seeing flips when do so, 

**/
inline float SGLM_Arcball::ZProject( const glm::vec2& p )
{
    float ll = glm::dot(p,p); 
    float z = ll <= 1.f ? std::sqrt(1.f - ll) : 0.f  ; 
    return z ; 
}

