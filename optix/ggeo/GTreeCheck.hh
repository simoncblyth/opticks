#pragma once

#include "stdlib.h"

class GGeo ; 
class GNode ; 

class GTreeCheck {
   public:
        GTreeCheck(GGeo* ggeo);
   public:
        void init();
        void traverse();
   private:
        void traverse( GNode* node, unsigned int depth );
   private:
       GGeo*                  m_ggeo ; 
       unsigned int           m_count ;  
 
};


inline GTreeCheck::GTreeCheck(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_count(0)
       {
          init();
       }




