#pragma once

#include "stdlib.h"

struct gfloat3 ; 
class GGeo ; 
class GNode ; 
//class GItemIndex ; 
class GColors ; 

class GBndLib ; 
class GSurfaceLib ; 


class GColorizer {
   public:
        typedef enum { SURFACE_INDEX, 
                       PSYCHEDELIC_VERTEX, 
                       PSYCHEDELIC_NODE, 
                       PSYCHEDELIC_MESH, 
                       NUM_STYLES } Style_t ;  
   public:
        GColorizer(GGeo* ggeo, GColorizer::Style_t style );  // vertex colorizer 
   private:
        void init(); 
   public:
        void setTarget(gfloat3* target);
        void setRepeatIndex(unsigned int ridx);

        //void setSurfaces(GItemIndex* surfaces);
        //void setMaterials(GItemIndex* materials);
        //void setColors(GColors* colors);
   public:
        gfloat3* make_color(unsigned int rgb);
   public:
        void traverse();   // full traverse from node 0, root
   private:
        gfloat3* getSurfaceColor(GNode* node);
        //gfloat3* getSurfaceColor_PRIOR(GNode* node);

        void traverse( GNode* node, unsigned int depth );
   private:
        gfloat3*               m_target ; 
        GGeo*                  m_ggeo ; 
        GBndLib*               m_blib ; 
        GSurfaceLib*           m_slib ; 
        GColors*               m_colors ; 

        Style_t                m_style ;  
        unsigned int           m_cur_vertices ;
        unsigned int           m_num_colorized ;
        //GItemIndex*            m_surfaces ;  
        //GItemIndex*            m_materials ;  
        unsigned int           m_repeat_index ; 
};


inline GColorizer::GColorizer(GGeo* ggeo, GColorizer::Style_t style ) 
       :
       m_target(NULL),
       m_ggeo(ggeo),
       m_blib(NULL),
       m_slib(NULL),
       m_colors(NULL),
       m_style(style),
       m_cur_vertices(0),
       m_num_colorized(0),
       //m_surfaces(NULL), 
       //m_materials(NULL),
       m_repeat_index(0)
{
     init();
}


inline void GColorizer::setTarget(gfloat3* target)
{
    m_target =  target ; 
}
//inline void GColorizer::setSurfaces(GItemIndex* surfaces)
//{
//    m_surfaces =  surfaces ; 
//}
//inline void GColorizer::setMaterials(GItemIndex* materials)
//{
//    m_materials = materials ; 
//}
inline void GColorizer::setRepeatIndex(unsigned int ridx)
{
    m_repeat_index = ridx ; 
}
//inline void GColorizer::setColors(GColors* colors)
//{
//    m_colors = colors ; 
//}


