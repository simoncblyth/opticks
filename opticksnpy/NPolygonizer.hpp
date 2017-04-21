#pragma once

#include "NPY_API_EXPORT.hh"


struct nnode ; 
struct nbbox ; 

class NParameters ; 
class NCSG ; 
class NTrianglesNPY ; 

class NPY_API NPolygonizer {
   public:
         NPolygonizer(NCSG* csg);
         NTrianglesNPY* polygonize();
   private:
         bool checkTris(NTrianglesNPY* tris);
         NTrianglesNPY* implicitMesher();
         NTrianglesNPY* dualContouringSample();
         NTrianglesNPY* marchingCubesNPY();
   private:
         NCSG*        m_csg ;
         nnode*       m_root ; 
         nbbox*       m_bbox ; 
         NParameters* m_meta ;  
         int          m_verbosity ; 
         int          m_index ; 
         const char*  m_poly ; 

};
