#pragma once

#include "NPY_API_EXPORT.hh"


struct nnode ; 
struct nbbox ; 
class NMeta ; 


class NCSG ; 
class NTrianglesNPY ; 


typedef enum 
{ 
   POLY_NONE, 
   POLY_MC, 
   POLY_DCS,
   POLY_IM,
   POLY_HY,
   POLY_BSP

}  NPolyMode_t ; 


class NPY_API NPolygonizer {
   public:
         static const char* POLY_NONE_ ; 
         static const char* POLY_MC_  ;
         static const char* POLY_DCS_ ;
         static const char* POLY_IM_ ;
         static const char* POLY_HY_ ;
         static const char* POLY_BSP_  ;

         static const char* PolyModeString(NPolyMode_t polymode);
         static  NPolyMode_t PolyMode(const char* poly);
   public:
         NPolygonizer(NCSG* csg);
         NTrianglesNPY* polygonize();
   private:
         bool checkTris(NTrianglesNPY* tris);
         NTrianglesNPY* implicitMesher();
         NTrianglesNPY* dualContouringSample();
         NTrianglesNPY* marchingCubesNPY();
         NTrianglesNPY* hybridMesher();
   private:
         NCSG*             m_csg ;
         nnode*            m_root ; 
         nbbox*            m_bbox ; 
         NMeta*            m_meta ;  

         int               m_verbosity ; 
         int               m_index ; 
         const char*       m_poly ; 
         NPolyMode_t       m_polymode ;  
         const char*       m_polycfg ; 

};
