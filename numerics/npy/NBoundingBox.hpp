#pragma once

#include <glm/glm.hpp>
#include <string>

class NBoundingBox {
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

inline const glm::vec4& NBoundingBox::getCenterExtent()
{
    return m_center_extent ; 
}


