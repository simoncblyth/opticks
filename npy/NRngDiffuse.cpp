
#include <sstream>

#include "BRng.hh"
#include "NRngDiffuse.hpp"
#include "NPY.hpp"


NRngDiffuse::NRngDiffuse(unsigned seed, float ctmin, float ctmax )
    :
    m_pi( glm::pi<float>() ),
    m_seed(seed),
    m_uazi(new BRng(0.f, 2.f*m_pi, seed, "uazi")),       // azimuthal
    m_upol(new BRng(-1.f, 1.f,   seed+1, "upol")),       // polar 
    m_unct(new BRng( ctmin, ctmax,   seed+2,  "unct"))   // costheta
{
}

std::string NRngDiffuse::desc() const 
{
    std::stringstream ss ; 

    ss << "NRngDiffuse"
       << " seed " << m_seed 
       << " uct.lo " << m_unct->getLo()
       << " uct.hi " << m_unct->getHi()
       ;

    return ss.str(); 
}


void NRngDiffuse::uniform_sphere(glm::vec4& u )
{
    float theta = m_uazi->one();
    float z = m_upol->one();
    float c = sqrtf(1.f - z*z );

    u.x = c*cosf(theta) ; 
    u.y = c*sinf(theta) ; 
    u.z = z ; 
    u.w = 0 ; 
} 


float NRngDiffuse::diffuse( glm::vec4& v, int& trials, const glm::vec3& dir )
{
    trials = 0 ; 
    float ndotv = 0.f ;
    do {
        uniform_sphere(v);
        ndotv = glm::dot(glm::vec3(v), dir);
        if (ndotv < 0.0f) 
        {   
            v = -v;
            ndotv = -ndotv;
        }   
        trials++ ; 
    } while (! (m_unct->one() < ndotv) );

    return ndotv ; 
}


NPY<float>* NRngDiffuse::uniform_sphere_sample(unsigned n)
{
    NPY<float>* buf = NPY<float>::make(n,4);
    buf->zero();

    glm::vec4 u ; 
    for(unsigned i=0 ; i < n ; i++)
    {
        uniform_sphere(u);
        buf->setQuad(u, i );
    } 
    return buf ; 
}


NPY<float>* NRngDiffuse::diffuse_sample(unsigned n, const glm::vec3& dir)
{
    NPY<float>* buf = NPY<float>::make(n,4);
    buf->zero();

    glm::vec4 u ; 
    int trials(0) ;
    for(unsigned i=0 ; i < n ; i++)
    {
        diffuse(u, trials, dir);
        u.w = float(trials) ; 

        buf->setQuad(u, i );
    } 
    return buf ; 
}


