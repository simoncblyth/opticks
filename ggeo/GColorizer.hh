#pragma once


#include "NQuad.hpp"

//class GGeo ; 
class GNode ; 
class OpticksColors ; 

class GBndLib ; 
class GSurfaceLib ; 
class GColors ; 

#include "GGEO_API_EXPORT.hh"
class GGEO_API GColorizer {
   public:
        typedef enum { SURFACE_INDEX, 
                       PSYCHEDELIC_VERTEX, 
                       PSYCHEDELIC_NODE, 
                       PSYCHEDELIC_MESH, 
                       NUM_STYLES } Style_t ;  
   public:
        GColorizer(GBndLib* blib, OpticksColors* colors, GColorizer::Style_t style ) ;
        //GColorizer(GGeo* ggeo, GColorizer::Style_t style );  // vertex colorizer 

        void writeVertexColors(GMergedMesh* mesh0, GSolid* root);
   private:
        void init(); 
   public:
        void setTarget(nvec3* target);  // where to write the colors
        void setRepeatIndex(unsigned int ridx);

   public:
        nvec3 make_color(unsigned int rgb);
   public:
        void traverse(GSolid* root);   // full recursive traverse from root
   private:
        nvec3 getSurfaceColor(GNode* node);

        void traverse_r( GNode* node, unsigned int depth );
   private:
        nvec3*                 m_target ; 
        //GGeo*                  m_ggeo ; 
        GBndLib*               m_blib ; 
        GSurfaceLib*           m_slib ; 
        OpticksColors*         m_colors ; 

        Style_t                m_style ;  
        unsigned int           m_cur_vertices ;
        unsigned int           m_num_colorized ;
        unsigned int           m_repeat_index ; 
};


