#include "Light.hh"

#include <glm/gtc/type_ptr.hpp>


glm::vec4 Light::getPosition()
{
    return glm::vec4(m_position.x, m_position.y, m_position.z,1.0f);
}   

float* Light::getPositionPtr()
{
    return glm::value_ptr(m_position);
}

glm::vec4 Light::getPosition(const glm::mat4& m2w)
{
    return m2w * getPosition();
} 



