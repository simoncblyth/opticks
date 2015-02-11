#include "GSkinSurface.hh"
#include "GPropertyMap.hh"

#include "stdlib.h"
#include "string.h"

GSkinSurface::GSkinSurface(const char* name) : 
    GPropertyMap(name, "skinsurface" ),
    m_skinsurface_vol(NULL) 
{
}

GSkinSurface::~GSkinSurface()
{
    free(m_skinsurface_vol);
}


void GSkinSurface::setSkinSurface(const char* vol)
{
    m_skinsurface_vol = strdup(vol);
}


char* GSkinSurface::getSkinSurfaceVol()
{
    return m_skinsurface_vol ; 
}

void GSkinSurface::Summary(const char* msg)
{
    if (m_skinsurface_vol)
    {
        printf("%s skinsurface vol %s \n", msg, m_skinsurface_vol );
    }
    else
    {
        printf("%s INCOMPLETE \n", msg );
    }
    GPropertyMap::Summary(msg);
}




