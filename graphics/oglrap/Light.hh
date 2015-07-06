#pragma once
#include <glm/glm.hpp>  

class Light {
   public:
       Light();
   public:
       glm::vec4 getPosition();
       glm::vec4 getDirection();
       float* getPositionPtr();
       float* getDirectionPtr();
       glm::vec4 getPosition(const glm::mat4& m2w);
       glm::vec4 getDirection(const glm::mat4& m2w);

   private:
       glm::vec3 m_position ; 
       glm::vec3 m_direction ; 

};


inline Light::Light()
{
}

