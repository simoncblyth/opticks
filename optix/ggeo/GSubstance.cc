#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "stdio.h"
#include "limits.h"
#include "assert.h"

// for persistency convenience would be better to deal in indices rather than pointers
//
GSubstance::GSubstance( GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface )
         : 
         m_imaterial(imaterial),
         m_omaterial(omaterial),
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
   if(m_omaterial)
   {
       char* pdig = m_omaterial->digest();
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





void GSubstance::Summary(const char* msg, unsigned int nline)
{
   assert(m_imaterial);

   char* dig = digest();
   char* imat = m_imaterial->getShortName("__dd__Materials__");
   char* omat = m_omaterial ? m_omaterial->getShortName("__dd__Materials__") : NULL  ;

   std::string imatk = m_imaterial ? m_imaterial->getKeysString() : "" ;
   std::string omatk = m_omaterial ? m_omaterial->getKeysString() : ""  ;
   std::string isurk = m_isurface ? m_isurface->getKeysString() : "" ; 
   std::string osurk = m_osurface ? m_osurface->getKeysString() : "" ; 

   char bmat[128];
   snprintf(bmat, 128,"imat/omat %s/%s", imat, omat );  
   printf("%s %4d [%s] %50s %s %s isur %s osur %s \n", msg, m_index, dig, bmat, imatk.c_str(), omatk.c_str(), isurk.c_str(), osurk.c_str() );

   if(m_imaterial) m_imaterial->Summary("imat", nline);
  // if(m_omaterial) m_omaterial->Summary("omat", nline);
  // if(m_isurface)  m_isurface->Summary("isurf", nline);
  // if(m_osurface)  m_osurface->Summary("osurf", nline);

   free(dig);
   free(imat);
   free(omat);
}



void GSubstance::setInnerMaterial(GPropertyMap* imaterial)
{
    m_imaterial = imaterial ; 
}
void GSubstance::setOuterMaterial(GPropertyMap* omaterial)
{
    m_omaterial = omaterial ; 
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
GPropertyMap* GSubstance::getOuterMaterial()
{
    return m_omaterial ; 
}
GPropertyMap* GSubstance::getInnerSurface()
{
    return m_isurface ; 
}
GPropertyMap* GSubstance::getOuterSurface()
{
    return m_osurface ; 
}



