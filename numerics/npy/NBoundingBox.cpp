#include "NBoundingBox.hpp"
#include <cfloat>
#include <algorithm>
#include <sstream>

#include "GLMFormat.hpp"

NBoundingBox::NBoundingBox()
   :
    m_low(FLT_MAX),
    m_high(-FLT_MAX)
{
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




