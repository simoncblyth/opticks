
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
    GPropertyMap<float>(name, index, "skinsurface", optical_surface ),
    m_skinsurface_vol(NULL)
{
    init();
}

void GSkinSurface::init()
{
    setStandardDomain( GDomain<float>::GetDefaultDomain()) ;   
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
    GPropertyMap<float>::Summary(msg, imod);


}




std::string GSkinSurface::description() const 
{
    std::stringstream ss ; 
    ss << "GSS:: " << GPropertyMap<float>::description() ; 
    return ss.str();
}



