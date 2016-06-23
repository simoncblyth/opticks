#pragma once



template<class T> class Counts ;

class GGeo ; 
class GBndLib ; 
class GMaterialLib ; 
class GNode ; 

#include "GGEO_API_EXPORT.hh"
class GGEO_API GTraverse {
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


