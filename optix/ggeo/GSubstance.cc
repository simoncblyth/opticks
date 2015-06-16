#include "GSubstance.hh"
#include "GPropertyMap.hh"
#include "md5digest.hh"
#include "GOpticalSurface.hh"

#include "stdio.h"
#include "limits.h"
#include "assert.h"



const char* GSubstance::imaterial = "imat" ;
const char* GSubstance::omaterial = "omat" ;
const char* GSubstance::isurface  = "isur" ;
const char* GSubstance::osurface  = "osur" ;
const char* GSubstance::iextra    = "iext" ;
const char* GSubstance::oextra    = "oext" ;

const char* GSubstance::inner_optical    = "iopt" ;
const char* GSubstance::outer_optical    = "oopt" ;





//
// How to handle optical_surface props ?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// * not so easy to add m_optical_surface due to GSubstance funny identity 
//   unless include it in the digest 
//
// * also it is associated with isurf or osurf, so better to attach there rather
//   than here ?
//
// * unless can finangle to arrange that GPropertyMap props are modified
//   based on optical_surface : then no need to change extant machinery 
//
//   but how to finangle ?  finish is obvious 
//  
//
//  OR rather the portion of OpticalSurface relevant to boundary identity 
//
//      name   
//             needs to be excluded : otherwise would get loadsa distinct boundaries 
//             all with the same optical properties
//
//      type   
//             no need to handle, as for Dayabay all surfaces are  dielectric_metal
//             but include in digest for future 
//
//      model  
//             (always unified) 
//             but include in digest for future 
//
//      finish 
//             (Dayabay: ground and polished occur)
//
//             can be expressed within GPropertyMap by 
//             branching REFLECTIVITY -> reflect_specular / reflect_diffuse
//
//      value  
//             in unified model this is SigmaAlpha
//             used to control FacetNormal 
//
//
//   simon:geant4.10.00.p01 blyth$ find source -name '*.cc' -exec grep -H SigmaAlpha {} \;
//   source/persistency/gdml/src/G4GDMLWriteSolids.cc:   G4double sval = (smodel==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();
//   source/processes/optical/src/G4OpBoundaryProcess.cc:       if (OpticalSurface) sigma_alpha = OpticalSurface->GetSigmaAlpha();
//
//
//
//


GSubstance::GSubstance()
         :
         m_imaterial(NULL),
         m_omaterial(NULL),
         m_isurface(NULL),
         m_osurface(NULL),
         m_iextra(NULL),
         m_oextra(NULL),
         m_index(UINT_MAX),
         m_inner_optical(NULL),
         m_outer_optical(NULL)
{
}

GSubstance::GSubstance( 
                 GPropertyMap<float>* imaterial, 
                 GPropertyMap<float>* omaterial, 
                 GPropertyMap<float>* isurface, 
                 GPropertyMap<float>* osurface, 
                 GPropertyMap<float>* iextra, 
                 GPropertyMap<float>* oextra,
                 GOpticalSurface* inner_optical,
                 GOpticalSurface* outer_optical
               )
         : 
         m_imaterial(imaterial),
         m_omaterial(omaterial),
         m_isurface(isurface),
         m_osurface(osurface),
         m_iextra(iextra),
         m_oextra(oextra),
         m_index(UINT_MAX),
         m_inner_optical(inner_optical),
         m_outer_optical(outer_optical)
{
}

GSubstance::~GSubstance()
{
}


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
   if(m_iextra)
   {
       char* pdig = m_iextra->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }
   if(m_oextra)
   {
       char* pdig = m_oextra->pdigest(ifr, ito);
       dig.update(pdig, strlen(pdig));
       free(pdig);
   }

   // TODO: inner_optical outer_optical

   return dig.finalize();
}




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
void GSubstance::setInnerExtra(GPropertyMap<float>* iextra)
{
    m_iextra = iextra ; 
}
void GSubstance::setOuterExtra(GPropertyMap<float>* oextra)
{
    m_oextra = oextra ; 
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
GPropertyMap<float>* GSubstance::getInnerExtra()
{
    return m_iextra ; 
}
GPropertyMap<float>* GSubstance::getOuterExtra()
{
    return m_oextra ; 
}



GPropertyMap<float>* GSubstance::getConstituentByIndex(unsigned int p)
{
    switch(p)
    {
       case 0:return m_imaterial ; break;
       case 1:return m_omaterial ; break;
       case 2:return m_isurface  ; break;
       case 3:return m_osurface  ; break;
       case 4:return m_iextra    ; break;
       case 5:return m_oextra    ; break;
    }
    return NULL ;
}

const char* GSubstance::getConstituentNameByIndex(unsigned int p)
{
    switch(p)
    {
       case 0:return imaterial ; break;
       case 1:return omaterial ; break;
       case 2:return isurface  ; break;
       case 3:return osurface  ; break;
       case 4:return iextra    ; break;
       case 5:return oextra    ; break;
    }
    return NULL ;
}



