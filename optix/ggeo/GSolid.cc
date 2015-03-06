#include "GSolid.hh"
#include "GPropertyMap.hh"

#include "stdio.h"

//
// hmm : maybe better to deal in material and surface indices rather than pointers ?
//
GSolid::GSolid( unsigned int index, GMatrixF* transform, GMesh* mesh, GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface )
         : 
         GNode(index),
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
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );

   if(m_imaterial) m_imaterial->Summary("imat");
   if(m_omaterial) m_omaterial->Summary("omat");
   if(m_isurface) m_isurface->Summary("isurf");
   if(m_osurface) m_osurface->Summary("osurf");
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



 
