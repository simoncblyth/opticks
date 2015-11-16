#pragma once
#include <cstring>
#include <glm/glm.hpp>

struct gbbox ; 

class GCache ; 
class GSolid ; 

class GTestShape {
   public:
       GTestShape(GCache* cache);
   public:
       static GSolid* make(char typecode, glm::vec4& spec );
       static GSolid* makeBox(glm::vec4& spec);
       static GSolid* makeSphere(glm::vec4& spec);
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


