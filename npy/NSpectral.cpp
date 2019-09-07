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



#include "GLMFormat.hpp"
#include "NCIE.hpp"
#include "NSpectral.hpp"
#include "PLOG.hh"


const char* NSpectral::sRGB_D65 = 
" 3.2404542 -1.5371385 -0.4985314 "
"-0.9692660  1.8760108  0.0415560 "
" 0.0556434 -0.2040259  1.0572252 "
;


NSpectral::NSpectral(unsigned int num_colors, unsigned int wlmin, unsigned int wlmax) 
    :
    m_num_colors(num_colors),
    m_wlmin(wlmin),
    m_wlmax(wlmax)
{
    init();
}


void NSpectral::init()
{ 
    m_XYZ2RGB = gmat3(sRGB_D65, false, " "); 
    LOG(info) << "NSpectral::init " << gformat(m_XYZ2RGB) ;  

    float step = float(m_wlmax - m_wlmin)/float(m_num_colors) ;
    float w0 = float(m_wlmin);

    for(unsigned int i=0 ; i < m_num_colors ; i++)
    {
        float w = w0 + step*i ; 
        unsigned int code = getColorCode(w) ;
        m_color_codes.push_back(code);
    }
}


const std::vector<unsigned int>& NSpectral::getColorCodes()
{
    return m_color_codes ; 
}


glm::vec3 NSpectral::getXYZ(float w)
{
    float X = ::cie_X(w);
    float Y = ::cie_Y(w);
    float Z = ::cie_Z(w);
    return glm::vec3(X,Y,Z) ; 
}


unsigned int NSpectral::getColorCode(float w)
{
    glm::vec3 raw = getXYZ(w);
    glm::vec3 scaled = raw/raw.y ; 
    glm::vec3 XYZ = glm::clamp(scaled, 0.f, 1.f );
    glm::vec3 RGB = m_XYZ2RGB * XYZ ; 

    LOG(info) << "NSpectral::getColorCode" 
              << " w " << std::setw(5)  << w 
              << " raw " << gformat(raw)
              << " scaled " << gformat(scaled)
              << " XYZ " << gformat(XYZ)
              << " RGB " << gformat(RGB)
              ;

    // huh not used ?
    return 0 ; 
}

void NSpectral::dump(const char* msg)
{
    LOG(info) << msg ; 

    for(VU_t::const_iterator it=m_color_codes.begin() ; it != m_color_codes.end() ; it++)
    {
        LOG(info) << *it ;    
    } 
}




