#include "GSubstance.hh"
#include "GPropertyMap.hh"

#include "stdio.h"
#include "limits.h"
#include "assert.h"

// for persistency convenience would be better to deal in indices rather than pointers
//
GSubstance::GSubstance()
         :
         m_imaterial(NULL),
         m_omaterial(NULL),
         m_isurface(NULL),
         m_osurface(NULL),
         m_index(UINT_MAX)
      //   m_texprops(NULL)
{
}

GSubstance::GSubstance( GPropertyMap<float>* imaterial, GPropertyMap<float>* omaterial, GPropertyMap<float>* isurface, GPropertyMap<float>* osurface )
         : 
         m_imaterial(imaterial),
         m_omaterial(omaterial),
         m_isurface(isurface),
         m_osurface(osurface),
         m_index(UINT_MAX)
    //     m_texprops(NULL)
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


/*
void GSubstance::setTexProps(GPropertyMap<float>* texprops)
{
    m_texprops = texprops ;
}
GPropertyMap<float>* GSubstance::getTexProps()
{
    return m_texprops ;
}
void GSubstance::dumpTexProps(const char* msg, double wavelength)
{
    printf("%s wavelength %10.3f \n", msg, wavelength );
    GPropertyMap<float>* ptex = getTexProps();
    assert(ptex);

    GDomain<float>* domain = ptex->getStandardDomain();
    unsigned int nprop = ptex->getNumProperties() ;
    assert(nprop % 4 == 0 );    
    assert(nprop == 16);

    const unsigned int nx = domain->getLength(); 
    const unsigned int ny = nprop/4 ; 
    assert(nx == 39 && ny == 4); 

    const char* mprop = "ridx/absl/sctl/reep" ;
    const char* sprop = "dete/abso/resp/redi" ;

    for( unsigned int line = 0; line < ny; ++line ) 
    {   
        unsigned int offset = line*ny ;   
        GPropertyD* p0 = ptex->getPropertyByIndex(offset+0) ;
        GPropertyD* p1 = ptex->getPropertyByIndex(offset+1) ;
        GPropertyD* p2 = ptex->getPropertyByIndex(offset+2) ;
        GPropertyD* p3 = ptex->getPropertyByIndex(offset+3) ;

        double v0 = p0->getInterpolatedValue(wavelength);
        double v1 = p1->getInterpolatedValue(wavelength);
        double v2 = p2->getInterpolatedValue(wavelength);
        double v3 = p3->getInterpolatedValue(wavelength);

        printf("GSubstance::dumpTexProps %10.3f nm line %u %s vals %13.4f %13.4f %13.4f %13.4f \n",
            wavelength,
            line,
            line < 2 ? mprop : sprop,
            v0,
            v1,
            v2,
            v3);
    }
}
*/


std::string GSubstance::getPDigestString(int ifr, int ito)
{
    return pdigest(ifr, ito);
}



char* GSubstance::pdigest(int ifr, int ito)
{
   MD5Digest dig ;

   if(m_imaterial)
   {
       char* pdig = m_imaterial->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }
   if(m_omaterial)
   {
       char* pdig = m_omaterial->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }
   if(m_isurface)
   {
       char* pdig = m_isurface->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }
   if(m_osurface)
   {
       char* pdig = m_osurface->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }
   return dig.finalize();
}


/*
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
*/




void GSubstance::Summary(const char* msg, unsigned int nline)
{
   assert(m_imaterial);

   char* dig = pdigest(0,4);
   char* imat = m_imaterial->getShortName("__dd__Materials__");
   char* omat = m_omaterial->getShortName("__dd__Materials__") ;
   char* isur = m_isurface ? m_isurface->getShortName("__dd__Geometry__") : (char*)"" ; 
   char* osur = m_osurface ? m_osurface->getShortName("__dd__Geometry__") : (char*)"" ; 

   /*
   std::string imatk = m_imaterial ? m_imaterial->getKeysString() : "" ;
   std::string omatk = m_omaterial ? m_omaterial->getKeysString() : ""  ;
   std::string isurk = m_isurface  ? m_isurface->getKeysString() : "" ; 
   std::string osurk = m_osurface  ? m_osurface->getKeysString() : "" ; 
   */

   char bmat[512];
   snprintf(bmat, 512,"%s/%s/%s/%s", imat, omat, isur, osur );  

   printf("%s : index %2u %s %s \n", msg, m_index, dig, bmat);  


   /*
   if(m_imaterial) m_imaterial->Summary("imat", nline);
   if(m_omaterial) m_omaterial->Summary("omat", nline);
   if(m_isurface)   m_isurface->Summary("isur", nline);
   if(m_osurface)   m_osurface->Summary("osur", nline);
   */
 
   free(dig);
   free(imat);
   free(omat);
   free(isur);
   free(osur);
}



void GSubstance::setInnerMaterial(GPropertyMap<float>* imaterial)
{
    m_imaterial = imaterial ; 
}
void GSubstance::setOuterMaterial(GPropertyMap<float>* omaterial)
{
    m_omaterial = omaterial ; 
}
void GSubstance::setInnerSurface(GPropertyMap<float>* isurface)
{
    m_isurface = isurface ; 
}
void GSubstance::setOuterSurface(GPropertyMap<float>* osurface)
{
    m_osurface = osurface ; 
}


GPropertyMap<float>* GSubstance::getInnerMaterial()
{
    return m_imaterial ; 
}
GPropertyMap<float>* GSubstance::getOuterMaterial()
{
    return m_omaterial ; 
}
GPropertyMap<float>* GSubstance::getInnerSurface()
{
    return m_isurface ; 
}
GPropertyMap<float>* GSubstance::getOuterSurface()
{
    return m_osurface ; 
}

GPropertyMap<float>* GSubstance::getConstituentByIndex(unsigned int p)
{
    switch(p)
    {
       case 0:return m_imaterial ; break;
       case 1:return m_omaterial ; break;
       case 2:return m_isurface  ; break;
       case 3:return m_osurface  ; break;
    }
    return NULL ;
}

