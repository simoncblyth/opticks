#include "GBorderSurface.hh"
#include "GPropertyMap.hh"

#include "stdlib.h"
#include "string.h"

GBorderSurface::GBorderSurface(const char* name, unsigned int index) : 
    GPropertyMap(name, index, "bordersurface" ),
    m_bordersurface_pv1(NULL),
    m_bordersurface_pv2(NULL)
{
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


char* GBorderSurface::getBorderSurfacePV1()
{
    return m_bordersurface_pv1 ; 
}
char* GBorderSurface::getBorderSurfacePV2()
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




void GBorderSurface::Summary(const char* msg)
{
    if( m_bordersurface_pv1 && m_bordersurface_pv2 )
    { 
        printf("%s bordersurface \n", msg  );
        printf("pv1 %s \n", m_bordersurface_pv1 );
        printf("pv2 %s \n", m_bordersurface_pv2 );
    }
    else
    {
        printf("%s INCOMPLETE %s \n", msg, getName() );
    }
    GPropertyMap::Summary(msg);
}




