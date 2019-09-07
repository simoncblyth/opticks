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

#include <cfloat>
#include <algorithm>
#include <sstream>

#include "GLMFormat.hpp"
#include "NBoundingBox.hpp"


NBoundingBox::NBoundingBox()
   :
    m_low(FLT_MAX),
    m_high(-FLT_MAX)
{
}    


const glm::vec4& NBoundingBox::getCenterExtent()
{
    return m_center_extent ; 
}


void NBoundingBox::update(const glm::vec3& low, const glm::vec3& high)
{
    m_low.x = std::min( m_low.x, low.x);
    m_low.y = std::min( m_low.y, low.y);
    m_low.z = std::min( m_low.z, low.z);

    m_high.x = std::max( m_high.x, high.x);
    m_high.y = std::max( m_high.y, high.y);
    m_high.z = std::max( m_high.z, high.z);

    m_center_extent.x = (m_low.x + m_high.x)/2.f ;
    m_center_extent.y = (m_low.y + m_high.y)/2.f ;
    m_center_extent.z = (m_low.z + m_high.z)/2.f ;
    m_center_extent.w = extent() ;
}

float NBoundingBox::extent()
{
   return extent(m_low, m_high);
}

float NBoundingBox::extent(const glm::vec3& low, const glm::vec3& high)
{
    glm::vec3 dim(high.x - low.x, high.y - low.y, high.z - low.z );
    float _extent(0.f) ;
    _extent = std::max( dim.x , _extent );
    _extent = std::max( dim.y , _extent );
    _extent = std::max( dim.z , _extent );
    _extent = _extent / 2.0f ;    
    return _extent ; 
}


std::string NBoundingBox::description()
{
    std::stringstream ss ; 

    ss << "NBoundingBox"
       << " low " << gformat(m_low)
       << " high " << gformat(m_high)
       << " ce " << gformat(m_center_extent)
       ;

    return ss.str();
}




