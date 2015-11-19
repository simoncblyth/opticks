#pragma once

template <typename T> class NPY ;
#include <cstddef>
#include <glm/glm.hpp>

struct gbbox ; 

class GCache ; 
class GGeo ; 
class GGeoLib ; 
class GBndLib ; 
class GSolid ; 

class GMaker {
   public:
       static const char* ZSPHERE ; 
       static const char* SPHERE ; 
       static const char* BOX ; 
       static const char* PMT ; 
       static const char* UNDEFINED ; 
   public:
       GMaker(GCache* cache);
   public:
       GSolid* make(unsigned int index, char shapecode, glm::vec4& param, const char* spec);
       static const char* ShapeName(char shapecode); 
   private:
       void init();    
       static GSolid* makeBox(glm::vec4& param);
       static GSolid* makeZSphere(glm::vec4& param);
       static GSolid* makeSphere(glm::vec4& param, unsigned int subdiv=3, const char* type="I");
       static GSolid* makeSphere(glm::vec4& param, NPY<float>* triangles);
       static GSolid* makeBox(gbbox& bbox);
   private:
       GCache*   m_cache ; 
       GGeo*     m_ggeo ; 
       GGeoLib*  m_geolib ; 
       GBndLib*  m_bndlib ; 
};


inline GMaker::GMaker(GCache* cache)
    :
    m_cache(cache),
    m_ggeo(NULL),
    m_geolib(NULL),
    m_bndlib(NULL)
{
    init();
}


