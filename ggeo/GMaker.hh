#pragma once

template <typename T> class NPY ;
class NTrianglesNPY ; 

#include <vector>
#include <cstddef>
#include <glm/fwd.hpp>

class Opticks ; 
#include "OpticksCSG.h"

struct gbbox ; 

class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GSolid ; 
class GMesh ; 

#include "GGEO_API_EXPORT.hh"
class GGEO_API GMaker {
   public:
       GMaker(Opticks* opticks, GGeo* ggeo=NULL);
   public:
       GSolid* make(unsigned int index, char nodecode, glm::vec4& param, const char* spec); // DONT USE THIS ONE IN NEW CODE
       GSolid* make(unsigned int index, OpticksCSG_t typecode, glm::vec4& param, const char* spec);
   private:
       void init();    
       static GSolid* makePrism(glm::vec4& param, const char* spec);
       static GSolid* makeBox(glm::vec4& param);
       static GSolid* makeZSphere(glm::vec4& param);
       static GSolid* makeZSphereIntersect(glm::vec4& param, const char* spec);
       static void makeBooleanComposite(char shapecode, std::vector<GSolid*>& solids,  glm::vec4& param, const char* spec);
   public:
       static GMesh* makeMarchingCubesTest();
   private:
       static GSolid* makeBox(gbbox& bbox);
   private:
       static GSolid* makeSubdivSphere(glm::vec4& param, unsigned int subdiv=3, const char* type="I");
       static NTrianglesNPY* makeSubdivSphere(unsigned int nsubdiv=3, const char* type="I");
       static GSolid* makeSphere(NTrianglesNPY* tris);
   private:
       Opticks*  m_opticks ; 
       GGeo*     m_ggeo ; 
       GGeoLib*  m_geolib ; 
       GBndLib*  m_bndlib ; 
};


