#include "GBorderSurface.hh"
#include "GPropertyMap.hh"

#include "stdlib.h"
#include "string.h"

GBorderSurface::GBorderSurface(const char* name) : 
    GPropertyMap(name, "bordersurface" ),
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




