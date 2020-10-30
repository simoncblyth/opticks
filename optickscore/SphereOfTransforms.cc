#include <sstream>

#include "PLOG.hh"
#include "NPY.hpp"
#include "NGLMExt.hpp"

#include "SphereOfTransforms.hh"

const plog::Severity  SphereOfTransforms::LEVEL = PLOG::EnvLevel("SphereOfTransforms", "DEBUG"); 


NPY<float>* SphereOfTransforms::Make(float radius, unsigned num_theta, unsigned num_phi) // static 
{
    SphereOfTransforms sot(radius, num_theta, num_phi); 
    return sot.getTransforms(); 
}


SphereOfTransforms::SphereOfTransforms(float radius, unsigned num_theta, unsigned num_phi)
    :
    m_radius(radius),
    m_num_theta(num_theta),
    m_num_phi(num_phi),
    m_num_transforms(2+(m_num_theta - 2)*m_num_phi),
    m_transforms(NPY<float>::make(m_num_transforms, 4,4))
{
    init(); 
}

std::string SphereOfTransforms::desc() const
{
    std::stringstream ss ; 
    ss 
       << "SphereOfTransforms"
       << " radius " << m_radius
       << " num_theta " << m_num_theta  
       << " num_phi " << m_num_phi  
       << " num_transforms " << m_num_transforms
       ;
    return ss.str(); 
}



NPY<float>* SphereOfTransforms::getTransforms() const 
{
    return m_transforms ; 
}

void SphereOfTransforms::init()
{
    m_transforms->zero(); 

    unsigned count(0);  
    for(unsigned itheta=0 ; itheta < m_num_theta ; itheta++)
    {
        float ftheta = float(itheta)/float(m_num_theta - 1) ; 
        bool is_pole = itheta == 0 || itheta == m_num_theta - 1 ; 

        for(unsigned iphi=0 ; iphi < m_num_phi ; iphi++)
        {
            if( is_pole && iphi > 0 ) break ;  // only the first phi slot for poles

            float fphi = float(iphi)/float(m_num_phi - 1) ; 

            glm::vec3 pos ; 
            glm::vec3 nrm ; 
            get_pos_nrm(pos, nrm, fphi, ftheta ); 
            
            glm::vec3 a(0.f, 0.f, 1.f);  // local frame reference "up" direction is +Z
            glm::vec3 b(-nrm);           // flip to inwards normal            

            // obtain the transform that orients the reference direction to the inwards normal direction
            // and then translates to the point in the sphere 
            glm::mat4 tr = nglmext::make_rotate_a2b_then_translate(a, b, pos );

            LOG(LEVEL)
                << " count " << count
                << " itheta " << itheta
                << " iphi " << iphi 
                << " tr " << glm::to_string( tr ) 
                ;

            m_transforms->setMat4(tr, count ); 
            count+= 1 ; 
        }
    }
    assert( count == m_num_transforms ); 
}

/**
SphereOfTransforms::get_pos_nrm
----------------------------------

pos
    point on the sphere
nrm 
    outwards normal : direction from origin to the point pos on the sphere

**/

void SphereOfTransforms::get_pos_nrm( glm::vec3& pos, glm::vec3& nrm, float fphi, float ftheta ) const
{
    pos.x = 0.f ; 
    pos.y = 0.f ; 
    pos.z = 0.f ; 

    nrm.x = 0.f ; 
    nrm.y = 0.f ; 
    nrm.z = 0.f ; 

    bool is_north_pole = ftheta == 0.f ; 
    bool is_south_pole = ftheta == 1.f ; 

    if(is_north_pole || is_south_pole) 
    {   
        pos += glm::vec3(0,0,is_north_pole ? m_radius : -m_radius ) ; 
        nrm.z = is_north_pole ? 1.f : -1.f ;    
    }   
    else
    {   
        const float pi = glm::pi<float>() ;

        float R = m_radius ; 
        float azimuth = 2.f*pi*fphi ; 
        float polar = pi*ftheta ;  

        float ca = cosf(azimuth);
        float sa = sinf(azimuth);
        float cp = cosf(polar);
        float sp = sinf(polar);

        pos += glm::vec3( R*ca*sp, R*sa*sp, R*cp );
 
        nrm.x = ca*sp ; 
        nrm.y = sa*sp ; 
        nrm.z = cp ; 
    }   
}


