#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "stdio.h"
#include "limits.h"
#include "assert.h"

// for persistency convenience would be better to deal in indices rather than pointers
//
GSubstance::GSubstance( GPropertyMap* imaterial, GPropertyMap* isurface, GPropertyMap* osurface )
         : 
         m_imaterial(imaterial),
         m_isurface(isurface),
         m_osurface(osurface),
         m_index(UINT_MAX)
{
}

GSubstance::~GSubstance()
{
}

void GSubstance::setIndex(unsigned int index)
{
    m_index = index ;
}

unsigned int GSubstance::getIndex()
{
    return m_index ;
}

char* GSubstance::digest()
{
   MD5Digest dig ;
   if(m_imaterial)
   {
       char* pdig = m_imaterial->digest();
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }  
   if(m_isurface)
   {
       char* pdig = m_isurface->digest();
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }  
   if(m_osurface)
   {
       char* pdig = m_osurface->digest();
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }  
   return dig.finalize();
}





void GSubstance::Summary(const char* msg )
{
   assert(m_imaterial);

   char* dig = digest();
   char* imat = m_imaterial->getShortName("__dd__Materials__");
   char* imatk = m_imaterial->getKeys() ;
 
   char* isurk = m_isurface ? m_isurface->getKeys() : NULL ; 
   char* osurk = m_osurface ? m_osurface->getKeys() : NULL ; 

   printf("%s %4d [%s] imat %30s %s isur %s osur %s \n", msg, m_index, dig, imat, imatk, isurk, osurk );


   /*
   if(m_imaterial) m_imaterial->Summary("imat");
   if(m_isurface) m_isurface->Summary("isurf");
   if(m_osurface) m_osurface->Summary("osurf");
    */

   free(dig);
   free(imat);
   free(imatk);
   free(isurk);
   free(osurk);
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
 
