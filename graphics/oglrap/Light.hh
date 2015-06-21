#pragma once
#include <glm/glm.hpp>  

class Light {
   public:
       Light();
   public:
       glm::vec4 getPosition();
       float* getPositionPtr();
       glm::vec4 getPosition(const glm::mat4& m2w);

   private:
       glm::vec3 m_position ; 

};


inline Light::Light()
{
}

