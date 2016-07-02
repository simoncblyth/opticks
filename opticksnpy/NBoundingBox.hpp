#pragma once

#include <string>
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NBoundingBox {
   public:
        NBoundingBox();
        void update(const glm::vec3& low, const glm::vec3& high);
        const glm::vec4& getCenterExtent();
   public:
        std::string description();
        static float extent(const glm::vec3& low, const glm::vec3& high);
        float extent();
   private:
       glm::vec3          m_low ; 
       glm::vec3          m_high ; 
       glm::vec4          m_center_extent ; 
};

#include "NPY_TAIL.hh"

