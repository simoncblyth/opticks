#pragma once

#include "NPY_API_EXPORT.hh"

class NCSG ; 
class NTrianglesNPY ; 

class NPY_API NPolygonizer {
   public:
         NPolygonizer(NCSG* csg);
         NTrianglesNPY* polygonize();
   private:
         NCSG* m_csg ; 

};
