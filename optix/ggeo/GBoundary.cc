#include "GBoundary.hh"
#include "GPropertyMap.hh"
#include "md5digest.hpp"

#include "stdio.h"
#include "limits.h"
#include "assert.h"



const char* GBoundary::imaterial = "imat" ;
const char* GBoundary::omaterial = "omat" ;
const char* GBoundary::isurface  = "isur" ;
const char* GBoundary::osurface  = "osur" ;
const char* GBoundary::iextra    = "iext" ;
const char* GBoundary::oextra    = "oext" ;


//
// How to handle optical_surface props ?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// * not so easy to add m_optical_surface due to GBoundary funny identity 
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


GBoundary::GBoundary()
         :
         m_imaterial(NULL),
         m_omaterial(NULL),
         m_isurface(NULL),
         m_osurface(NULL),
         m_iextra(NULL),
         m_oextra(NULL),
         m_index(UINT_MAX)
{
}

GBoundary::GBoundary( 
                 GPropertyMap<float>* imaterial, 
                 GPropertyMap<float>* omaterial, 
                 GPropertyMap<float>* isurface, 
                 GPropertyMap<float>* osurface, 
                 GPropertyMap<float>* iextra, 
                 GPropertyMap<float>* oextra
               )
         : 
         m_imaterial(imaterial),
         m_omaterial(omaterial),
         m_isurface(isurface),
         m_osurface(osurface),
         m_iextra(iextra),
         m_oextra(oextra),
         m_index(UINT_MAX)
{
}

GBoundary::~GBoundary()
{
}


std::string GBoundary::getPDigestString(int ifr, int ito)
{
    return pdigest(ifr, ito);
}



char* GBoundary::pdigest(int ifr, int ito)
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


   return dig.finalize();
}




void GBoundary::Summary(const char* msg, unsigned int nline)
{
   assert(m_imaterial);

   char* dig = pdigest(0,4);
   const char* imat = m_imaterial->getShortName();
   const char* omat = m_omaterial->getShortName() ;
   const char* isur = m_isurface ? m_isurface->getShortName() : "" ; 
   const char* osur = m_osurface ? m_osurface->getShortName() : "" ; 

   char bmat[512];
   snprintf(bmat, 512,"%s/%s/%s/%s", imat, omat, isur, osur );  

   printf("%s : index %2u %s %s \n", msg, m_index, dig, bmat);  

   free(dig);

   //free(imat);
   //free(omat);
   //free(isur);
   //free(osur);
}



GPropertyMap<float>* GBoundary::getConstituentByIndex(unsigned int p)
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

const char* GBoundary::getConstituentNameByIndex(unsigned int p)
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



