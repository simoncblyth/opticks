#pragma once
#include <cstring>
#include <glm/glm.hpp>

struct gbbox ; 

class GCache ; 
class GSolid ; 

class GTestShape {
   public:
       static const char* SPHERE ; 
       static const char* BOX ; 
       static const char* PMT ; 
       static const char* UNDEFINED ; 
   public:
       GTestShape(GCache* cache);
   public:
       static const char* ShapeName(char shapecode); 
       static GSolid* make(char shapecode, glm::vec4& spec );
       static GSolid* makeBox(glm::vec4& spec);
       static GSolid* makeSphere(glm::vec4& spec, unsigned int subdiv=3);
   private:
       static GSolid* makeBox(gbbox& bbox);
   private:
       GCache*  m_cache ; 

};


inline GTestShape::GTestShape(GCache* cache)
    :
    m_cache(cache)
{
}


