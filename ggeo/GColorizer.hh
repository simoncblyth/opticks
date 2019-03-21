#pragma once


#include "NQuad.hpp"
#include "plog/Severity.h"

class GNode ; 
class OpticksColors ; 

class GNodeLib ; 
class GGeoLib ; 
class GBndLib ; 
class GSurfaceLib ; 
class GColors ; 

/**
GColorizer
============

Canonical m_colorizer instances are residents of GGeo and GScene, 
For GGeo are instanciated in GGeo::init pre-cache.

Instanciation just holds on to constituent pointers 
the action happens when writeVertexColors() is called. This 
gets invoked by GGeo::prepareVertexColors.


GColorizer::traverse
----------------------

Visits all vertices of selected volumes setting the 
vertex colors of the GMergedMesh based on indices of
objects configured via the style.


Flipping normals and visibility
----------------------------------

Observations as change gl/nrm/vert.glsl and vertex colors

* initially without flipping normals had to fine tune light positions
  in order to see anything and everything was generally very dark

* with normals flipped things become much more visible, which makes
  sense given the realisation that the "canonical" lighting situation 
  is from inside geometry, which togther with outwards normals
  means that have to flip the normals to see something 
     

**/

#include "GGEO_API_EXPORT.hh"
class GGEO_API GColorizer {
   public:
        static const plog::Severity LEVEL ; 
   public:
        typedef enum { SURFACE_INDEX, 
                       PSYCHEDELIC_VERTEX, 
                       PSYCHEDELIC_NODE, 
                       PSYCHEDELIC_MESH, 
                       NUM_STYLES } Style_t ;  
   public:
        GColorizer(GNodeLib* nodelib, GGeoLib* geolib, GBndLib* blib, OpticksColors* colors, GColorizer::Style_t style ) ;
        void writeVertexColors();
   private:
        void init(); 
        void writeVertexColors(GMergedMesh* mesh0, GVolume* root);
   public:
        void setTarget(nvec3* target);  // where to write the colors
        void setRepeatIndex(unsigned int ridx);

   public:
        nvec3 make_color(unsigned int rgb);
   public:
        void traverse(GVolume* root);   // full recursive traverse from root
   private:
        nvec3 getSurfaceColor(GNode* node);

        void traverse_r( GNode* node, unsigned int depth );
   private:
        nvec3*                 m_target ; 
        GNodeLib*              m_nodelib ; 
        GGeoLib*               m_geolib ; 
        GBndLib*               m_blib ; 
        GSurfaceLib*           m_slib ; 
        OpticksColors*         m_colors ; 

        Style_t                m_style ;  
        unsigned int           m_cur_vertices ;
        unsigned int           m_num_colorized ;
        unsigned int           m_repeat_index ; 
};


