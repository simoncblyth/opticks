#pragma once
#include <functional>
#include <glm/glm.hpp>

struct FGLite 
{
    std::function<float(float,float,float)>* func ; 

    int       resolution ; 
    float     elem_offset ;  // -0.5f  is a centered choice

    glm::vec3 elem ;         // grid element size
    glm::vec3 min ; 
    glm::vec3 max ; 
    glm::ivec3 offset ; 

    // floated ijk grid coordinates to world coordinates
    glm::vec3 position_f(const glm::vec3& ijkf_ ) const 
    {

        glm::vec3 ijkf = ijkf_ ;  // -64:63
        ijkf -= offset ;          //   0:127 
        //ijkf += elem_offset ;     //   -0.5:126.5
        ijkf /= resolution ;    // resolution - 1 ? << match NGrid bug 

         // trying to mimick the NFieldGrid appoach 

        glm::vec3 world ;
        world.x = min.x + ijkf.x * (max.x - min.x ) ;
        world.y = min.y + ijkf.y * (max.y - min.y ) ;
        world.z = min.z + ijkf.z * (max.z - min.z ) ;
 
        return world ; 
    }

    // floated ijk grid coordinates to implicit function value
    float value_f(const glm::vec3& ijkf) const 
    {
        glm::vec3 world = position_f(ijkf);
        float val = (*func)(world.x, world.y, world.z); 
        return val ; 
    } 

};



