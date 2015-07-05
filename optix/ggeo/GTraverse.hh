#pragma once

#include "stdlib.h"


template<class T> class Counts ;

class GGeo ; 
class GNode ; 

class GTraverse {
   public:
        GTraverse(GGeo* ggeo);
   public:
        void init();
        void traverse();
   private:
        void traverse( GNode* node, unsigned int depth );
   private:
       GGeo*                  m_ggeo ; 
       Counts<unsigned int>*  m_materials_count ; 
 
};


inline GTraverse::GTraverse(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_materials_count(NULL)
       {
          init();
       }




