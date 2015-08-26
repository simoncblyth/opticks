#pragma once

#include <string>
#include <vector>

class GGeo ; 
class GNode ; 

class GTreePresent {

   // cf env/geant4/geometry/collada/g4daenode.py DAESubTree

   public:
        GTreePresent(GGeo* ggeo, unsigned int top, unsigned int depth_max, unsigned int sibling_max );
   public:
        void traverse();
        void dump(const char* msg="GTreePresent::dump");
        void write(const char* path);
   private:
        void traverse( GNode* node, unsigned int depth, unsigned int numSibling, unsigned int siblingIndex);
   private:
       GGeo*                    m_ggeo ; 
       unsigned int             m_top ; 
       unsigned int             m_depth_max ; 
       unsigned int             m_sibling_max ; 
   private:
       std::vector<std::string> m_flat ; 
 
};


inline GTreePresent::GTreePresent(GGeo* ggeo, unsigned int top, unsigned int depth_max, unsigned int sibling_max) 
       :
       m_ggeo(ggeo),
       m_top(top),
       m_depth_max(depth_max),
       m_sibling_max(sibling_max)
       {
       }




