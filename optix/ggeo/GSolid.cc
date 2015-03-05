#include "GSolid.hh"

#include "stdio.h"

GSolid::GSolid( GMatrixF* transform, GMesh* mesh, GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface )
         : 
         m_transform(transform),
         m_mesh(mesh),
         m_imaterial(imaterial),
         m_omaterial(omaterial),
         m_isurface(isurface),
         m_osurface(osurface)
{
    // NB not taking ownership yet 
}

GSolid::~GSolid()
{
}

void GSolid::Summary(const char* msg )
{
   printf("%s\n", msg );
}



void GSolid::setInnerMaterial(GPropertyMap* imaterial)
{
    m_imaterial = imaterial ; 
}
void GSolid::setOuterMaterial(GPropertyMap* omaterial)
{
    m_omaterial = omaterial ; 
}

void GSolid::setInnerSurface(GPropertyMap* isurface)
{
    m_isurface = isurface ; 
}
void GSolid::setOuterSurface(GPropertyMap* osurface)
{
    m_osurface = osurface ; 
}

GPropertyMap* GSolid::getInnerMaterial()
{
    return m_imaterial ; 
}
GPropertyMap* GSolid::getOuterMaterial()
{
    return m_omaterial ; 
}
GPropertyMap* GSolid::getInnerSurface()
{
    return m_isurface ; 
}
GPropertyMap* GSolid::getOuterSurface()
{
    return m_osurface ; 
}



 
