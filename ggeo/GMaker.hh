#pragma once

template <typename T> class NPY ;
class NTrianglesNPY ; 

#include <vector>
#include <cstddef>
#include <glm/fwd.hpp>

class Opticks ; 
#include "OpticksShape.h"

struct gbbox ; 

class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GSolid ; 

#include "GGEO_API_EXPORT.hh"
class GGEO_API GMaker {
   public:
       static const char* ZSPHERE ; 
       static const char* ZLENS ; 
       static const char* SPHERE ; 
       static const char* BOX ; 
       static const char* PMT ; 
       static const char* PRISM ; 
       static const char* UNION ; 
       static const char* INTERSECTION ; 
       static const char* DIFFERENCE ; 
       static const char* UNDEFINED ; 
       static OpticksShape_t NodeFlag(char nodecode);
       static const char* NodeName(char nodecode); 
       static char NodeCode(const char* nodename); 
       static bool IsBooleanShape(char nodecode); 
       static bool IsCompositeShape(char nodecode); 
       static unsigned BooleanFlag(char nodecode); 
   public:
       GMaker(Opticks* opticks, GGeo* ggeo=NULL);
   public:
       GSolid* make(unsigned int index, char nodecode, glm::vec4& param, const char* spec);
   private:
       void init();    
       static GSolid* makePrism(glm::vec4& param, const char* spec);
       static GSolid* makeBox(glm::vec4& param);
       static GSolid* makeZSphere(glm::vec4& param);
       static GSolid* makeZSphereIntersect(glm::vec4& param, const char* spec);
       static void makeBooleanComposite(char shapecode, std::vector<GSolid*>& solids,  glm::vec4& param, const char* spec);
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


