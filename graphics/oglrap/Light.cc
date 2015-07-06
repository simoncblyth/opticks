#include "Light.hh"

#include <glm/gtc/type_ptr.hpp>


glm::vec4 Light::getPosition()
{
    return glm::vec4(m_position.x, m_position.y, m_position.z,1.0f);
}   
glm::vec4 Light::getDirection()
{
    return glm::vec4(m_direction.x, m_direction.y, m_direction.z,0.0f);
}   



float* Light::getPositionPtr()
{
    return glm::value_ptr(m_position);
}
float* Light::getDirectionPtr()
{
    return glm::value_ptr(m_direction);
}


glm::vec4 Light::getPosition(const glm::mat4& m2w)
{
    return m2w * getPosition();
} 
glm::vec4 Light::getDirection(const glm::mat4& m2w)
{
    return m2w * getDirection();
} 




