#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "stdio.h"

// for persistency convenience would be better to deal in indices rather than pointers
//
GSubstance::GSubstance( GPropertyMap* imaterial, GPropertyMap* isurface, GPropertyMap* osurface )
         : 
         m_imaterial(imaterial),
         m_isurface(isurface),
         m_osurface(osurface)
{
}

GSubstance::~GSubstance()
{
}



void GSubstance::Summary(const char* msg )
{
   printf("%s\n", msg );

   if(m_imaterial) m_imaterial->Summary("imat");
   if(m_isurface) m_isurface->Summary("isurf");
   if(m_osurface) m_osurface->Summary("osurf");
}



void GSubstance::setInnerMaterial(GPropertyMap* imaterial)
{
    m_imaterial = imaterial ; 
}
void GSubstance::setInnerSurface(GPropertyMap* isurface)
{
    m_isurface = isurface ; 
}
void GSubstance::setOuterSurface(GPropertyMap* osurface)
{
    m_osurface = osurface ; 
}


GPropertyMap* GSubstance::getInnerMaterial()
{
    return m_imaterial ; 
}
GPropertyMap* GSubstance::getInnerSurface()
{
    return m_isurface ; 
}
GPropertyMap* GSubstance::getOuterSurface()
{
    return m_osurface ; 
}



bool GSubstance::matches(GSubstance* other)
{
    return 
           getInnerMaterial() == other->getInnerMaterial() 
           && 
           getInnerSurface() == other->getInnerSurface() 
           &&
           getOuterSurface() == other->getOuterSurface() 
           ;
}
 
