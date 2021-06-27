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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sstream>


#include "GOpticalSurface.hh"
#include "GSkinSurface.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GDomain.hh"

#include "GGEO_BODY.hh"


GSkinSurface::GSkinSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface) 
   : 
    GPropertyMap<double>(name, index, "skinsurface", optical_surface ),
    m_skinsurface_vol(NULL)
{
    init();
}

void GSkinSurface::init()
{
    setStandardDomain( GDomain<double>::GetDefaultDomain()) ;   
    // ensure the domain is set before adding properties, like AssimpGGeo 
}


GSkinSurface::~GSkinSurface()
{
}


void GSkinSurface::setSkinSurface(const char* vol)
{
    m_skinsurface_vol = strdup(vol);
}


const char* GSkinSurface::getSkinSurfaceVol() const 
{
    return m_skinsurface_vol ; 
}


/*
bool GSkinSurface::matches(const char* lv) const 
{
    return strncmp(m_skinsurface_vol, lv, strlen(lv)) == 0; 
}
*/

bool GSkinSurface::matches(const char* lv) const 
{
    return strcmp(m_skinsurface_vol, lv) == 0; 
}



void GSkinSurface::Summary(const char* msg, unsigned int imod) const 
{
    if (m_skinsurface_vol)
    {
       //printf("%s skinsurface vol %s \n", msg, m_skinsurface_vol );
    }
    else
    {
        printf("%s INCOMPLETE \n", msg );
    }
    GPropertyMap<double>::Summary(msg, imod);


}




std::string GSkinSurface::description() const 
{
    std::stringstream ss ; 
    ss << "GSS:: " << GPropertyMap<double>::description() ; 
    return ss.str();
}



