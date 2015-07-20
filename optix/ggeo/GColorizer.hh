#pragma once

#include "stdlib.h"

struct gfloat3 ; 
class GGeo ; 
class GNode ; 
class GItemIndex ; 

class GColorizer {
   public:
        GColorizer(gfloat3* target, GGeo* ggeo );
   public:
        void setRepeatIndex(unsigned int ridx);
        void setSurfaces(GItemIndex* surfaces);
        void setMaterials(GItemIndex* materials);
   public:
        gfloat3* make_color(unsigned int rgb);
   public:
        void traverse();
   private:
        gfloat3* getSurfaceColor(GNode* node);
        void traverse( GNode* node, unsigned int depth );
   private:
        GGeo*                  m_ggeo ; 
        gfloat3*               m_target ; 
        unsigned int           m_cur_vertices ;
        unsigned int           m_num_colorized ;
        GItemIndex*            m_surfaces ;  
        GItemIndex*            m_materials ;  
        unsigned int           m_repeat_index ; 
};


inline GColorizer::GColorizer(gfloat3* target, GGeo* ggeo ) 
       :
       m_target(target),
       m_ggeo(ggeo),
       m_cur_vertices(0),
       m_num_colorized(0),
       m_surfaces(NULL), 
       m_materials(NULL),
       m_repeat_index(0)
{
}

inline void GColorizer::setSurfaces(GItemIndex* surfaces)
{
    m_surfaces =  surfaces ; 
}
inline void GColorizer::setMaterials(GItemIndex* materials)
{
    m_materials = materials ; 
}
inline void GColorizer::setRepeatIndex(unsigned int ridx)
{
    m_repeat_index = ridx ; 
}




