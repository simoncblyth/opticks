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
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "GBorderSurface.hh"
#include "GOpticalSurface.hh"
#include "GPropertyMap.hh"
#include "GDomain.hh"

#include "GGEO_BODY.hh"



GBorderSurface::GBorderSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface ) : 
    GPropertyMap<double>(name, index, "bordersurface", optical_surface ),
    m_bordersurface_pv1(NULL),
    m_bordersurface_pv2(NULL)
{
    init() ;
}

void GBorderSurface::init()
{
    setStandardDomain( GDomain<double>::GetDefaultDomain()) ;   
    // ensure the domain is set before adding properties, like AssimpGGeo 
}


GBorderSurface::~GBorderSurface()
{
    free(m_bordersurface_pv1);
    free(m_bordersurface_pv2);
}

void GBorderSurface::setBorderSurface(const char* pv1, const char* pv2)
{
    m_bordersurface_pv1 = strdup(pv1);
    m_bordersurface_pv2 = strdup(pv2);
}


char* GBorderSurface::getPV1() const 
{
    return m_bordersurface_pv1 ; 
}
char* GBorderSurface::getPV2() const 
{
    return m_bordersurface_pv2 ; 
}


bool GBorderSurface::matches(const char* pv1, const char* pv2)
{
    return 
          strncmp(m_bordersurface_pv1, pv1, strlen(pv1)) == 0  
       && strncmp(m_bordersurface_pv2, pv2, strlen(pv2)) == 0   ;
}

bool GBorderSurface::matches_swapped(const char* pv1, const char* pv2)
{
    return 
          strncmp(m_bordersurface_pv2, pv1, strlen(pv1)) == 0  
       && strncmp(m_bordersurface_pv1, pv2, strlen(pv2)) == 0   ;
}

bool GBorderSurface::matches_either(const char* pv1, const char* pv2)
{
    return matches(pv1, pv2) || matches_swapped(pv1, pv2) ; 
}

bool GBorderSurface::matches_one(const char* pv1, const char* pv2)
{
    return 
          strncmp(m_bordersurface_pv1, pv1, strlen(pv1)) == 0  
       || strncmp(m_bordersurface_pv2, pv2, strlen(pv2)) == 0   ;
}




void GBorderSurface::Summary(const char* msg, unsigned int imod)
{
    if( m_bordersurface_pv1 && m_bordersurface_pv2 )
    { 
        //printf("%s bordersurface \n", msg  );
        //printf("pv1 %s \n", m_bordersurface_pv1 );
        //printf("pv2 %s \n", m_bordersurface_pv2 );
    }
    else
    {
        printf("%s INCOMPLETE %s \n", msg, getName() );
    }
    GPropertyMap<double>::Summary(msg, imod);
}



std::string GBorderSurface::description()
{
    std::stringstream ss ; 
    ss << "GBS:: " << GPropertyMap<double>::description() ; 
    return ss.str();
}


std::string GBorderSurface::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "GBS:: " 
       << " " << std::setw(90) << getName()
       << " pv1: " << std::setw(40) << getPV1()
       << " pv2: " << std::setw(40) << getPV2()
       ;
    return ss.str();
}



