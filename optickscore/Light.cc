/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include "NGLM.hpp"
#include "Light.hh"


Light::Light()
   :
   m_position(0.f,0.f,0.f),
   m_direction(1.f,0.f,0.f)
{
}


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




