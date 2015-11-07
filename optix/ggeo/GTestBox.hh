#pragma once
#include <cstring>

#include "GVector.hh"
class GCache ; 
class GMesh ; 
class GSolid ; 

class GTestBox {
   public:
       enum { NUM_VERTICES = 24, 
              NUM_FACES = 6*2 } ;
   public:
       GTestBox(GCache* cache);
   public:
       static GMesh*  makeMesh(gbbox& bbox, unsigned int meshindex);
       static GSolid* makeSolid(gbbox& bbox, unsigned int meshindex, unsigned int nodeindex);
   private:
       GCache*  m_cache ; 

};


inline GTestBox::GTestBox(GCache* cache)
    :
    m_cache(cache)
{
}

