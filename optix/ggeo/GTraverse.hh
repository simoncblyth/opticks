#pragma once

#include "stdlib.h"


template<class T> class Counts ;

class GGeo ; 
class GBndLib ; 
class GMaterialLib ; 
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
       GBndLib*               m_blib ; 
       GMaterialLib*          m_mlib ; 
       Counts<unsigned int>*  m_materials_count ; 
 
};


inline GTraverse::GTraverse(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_blib(NULL),
       m_mlib(NULL),
       m_materials_count(NULL)
       {
          init();
       }




