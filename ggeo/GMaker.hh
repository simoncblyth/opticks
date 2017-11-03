#pragma once

template <typename T> class NPY ;
class NTrianglesNPY ; 

#include <vector>
#include <cstddef>
#include <glm/fwd.hpp>

class Opticks ; 
#include "OpticksCSG.h"

class NCSG ; 
struct gbbox ; 
struct nnode ; 

class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GSolid ; 
class GMesh ; 

/**

GMaker
=======

Principal instances

* m_maker member of GGeoTest 


**/

#include "GGEO_API_EXPORT.hh"
class GGEO_API GMaker {
       friend class GMakerTest ; 
    public:
        static std::string PVName(const char* shapename, int idx=-1);
        static std::string LVName(const char* shapename, int idx=-1);
   public:
       GMaker(Opticks* opticks, GGeo* ggeo=NULL);
   public:
       GSolid* make(unsigned int index, OpticksCSG_t typecode, glm::vec4& param, const char* spec);
       GSolid* makeFromCSG(NCSG* csg, unsigned verbosity );
   private:
       void init();    

       static GSolid* makeFromCSG(NCSG* csg, GBndLib* bndlib, unsigned verbosity );
       static GSolid* makePrism(glm::vec4& param, const char* spec);
       static GSolid* makeBox(glm::vec4& param);
       static GSolid* makeZSphere(glm::vec4& param);
       static GSolid* makeZSphereIntersect_DEAD(glm::vec4& param, const char* spec);
       static void makeBooleanComposite(char shapecode, std::vector<GSolid*>& solids,  glm::vec4& param, const char* spec);
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


