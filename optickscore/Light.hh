#pragma once

#include <glm/fwd.hpp>  
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API Light {
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

#include "OKCORE_TAIL.hh"

