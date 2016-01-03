#pragma once

#include <vector>
#include <glm/glm.hpp>

class NSpectral {
    public:
       static const char* sRGB_D65 ;
    public:
        NSpectral(unsigned int num_colors=100, unsigned int wlmin=380, unsigned int wlmax=780);
    public:
        const std::vector<unsigned int>& getColorCodes();
    private:
        void init();
        glm::vec3 getXYZ(float wavelength);
        unsigned int getColorCode(float wavelength);
    private:
        unsigned int m_num_colors ; 
        unsigned int m_wlmin ; 
        unsigned int m_wlmax ; 
        glm::mat3    m_XYZ2RGB ; 
        std::vector<unsigned int> m_color_codes ; 
};


inline NSpectral::NSpectral(unsigned int num_colors, unsigned int wlmin, unsigned int wlmax) 
    :
    m_num_colors(num_colors),
    m_wlmin(wlmin),
    m_wlmax(wlmax)
{
    init();
}

const std::vector<unsigned int>& NSpectral::getColorCodes()
{
    return m_color_codes ; 
}



