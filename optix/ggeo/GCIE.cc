#include "GCIE.hh"
#include "GAry.hh"

#include "GLMFormat.hpp"
#include "NLog.hpp"

const char* GCIE::sRGB_D65 = 
" 3.2404542 -1.5371385 -0.4985314 "
"-0.9692660  1.8760108  0.0415560 "
" 0.0556434 -0.2040259  1.0572252 "
;
void GCIE::init()
{
    m_XYZ2RGB = gmat3(sRGB_D65, true, " "); 

    m_nm = GAry<float>::from_domain(m_domain);
    m_X = GAry<float>::cie_X(m_nm);
    m_Y = GAry<float>::cie_Y(m_nm);
    m_Z = GAry<float>::cie_Z(m_nm);
}

void GCIE::dump(const char* msg)
{
    LOG(info) << msg ; 
    m_nm->Summary("nm");
    m_X->Summary("X");
    m_Y->Summary("Y");
    m_Z->Summary("Z");


    float* nm = m_nm->getValues();
    unsigned int len = m_nm->getLength();

    for(unsigned int i=0 ; i < len ; i++)
    {
        float wavelength = nm[i] ;
        glm::vec3 XYZ = getXYZ(wavelength);
        glm::vec3 RGB = getRGB(wavelength);
        LOG(info) << std::setw(5) << wavelength 
                  << " XYZ " << gformat(XYZ) 
                  << " RGB " << gformat(RGB) 
                  ;
    }

    LOG(info) << "XMax " << getXMax() ;
    LOG(info) << "YMax " << getYMax() ;
    LOG(info) << "ZMax " << getZMax() ;

} 

float GCIE::getInterpolatedX(float wavelength)
{
    return GAry<float>::np_interp( wavelength, m_nm, m_X );
}
float GCIE::getInterpolatedY(float wavelength)
{
    return GAry<float>::np_interp( wavelength, m_nm, m_Y );
}
float GCIE::getInterpolatedZ(float wavelength)
{
    return GAry<float>::np_interp( wavelength, m_nm, m_Z );
}


glm::vec3 GCIE::getXYZ(float wavelength)
{
    return glm::vec3( 
        getInterpolatedX(wavelength), 
        getInterpolatedY(wavelength), 
        getInterpolatedZ(wavelength)) ;
}

float GCIE::getXMax()
{
    unsigned int idx(0) ;
    return m_X->max(idx); 
}

float GCIE::getYMax()
{
    unsigned int idx(0) ;
    return m_Y->max(idx); 
}

float GCIE::getZMax()
{
    unsigned int idx(0) ;
    return m_Z->max(idx); 
}

glm::vec3 GCIE::getRGB(float wavelength)
{
    float scale = getYMax();
    glm::vec3 raw = getXYZ(wavelength);
    glm::vec3 scaled = raw/scale ; 
    glm::vec3 rgb0 = m_XYZ2RGB * scaled ; 
    glm::vec3 rgb1 = glm::clamp(rgb0, 0.f, 1.f ); 
    return rgb1 ; 
}
 

