#include "GSkinSurface.hh"
#include "GOpticalSurface.hh"
#include "GPropertyMap.hh"

#include "stdlib.h"
#include "string.h"

GSkinSurface::GSkinSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface) : 
    GPropertyMap<float>(name, index, "skinsurface" ),
    m_optical_surface(new GOpticalSurface(optical_surface)),
    m_skinsurface_vol(NULL)
{
}

GSkinSurface::~GSkinSurface()
{
    free(m_skinsurface_vol);
    delete m_optical_surface ; 
}


void GSkinSurface::setSkinSurface(const char* vol)
{
    m_skinsurface_vol = strdup(vol);
}


char* GSkinSurface::getSkinSurfaceVol()
{
    return m_skinsurface_vol ; 
}

bool GSkinSurface::matches(const char* lv)
{
    return strncmp(m_skinsurface_vol, lv, strlen(lv)) == 0; 
}


void GSkinSurface::Summary(const char* msg, unsigned int imod)
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


    if(m_optical_surface) m_optical_surface->Summary(msg, imod);
}




