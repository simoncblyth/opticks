/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

Canonical m_colorizer instance is resident of GGeo 
and is pre-cache instanciated in GGeo::init.

Instanciation just holds on to constituent pointers 
the action happens when writeVertexColors() is called. This 
gets invoked by GGeo::prepareVertexColors.


Seeing the effects in the GUI
-------------------------------

Use "E" key to switch the Composition::nextGeometryStyle.  
Looks like "geo:vtxcol" is showing the results of GColorizer.

geo:default
   looks to be light influenced, but appears very dark
geo:nrmcol
    normal shader vibrant colors for both instances and remainder volumes
geo:vtxcol
    flat colors varying for remainder volumes, all instances are mid grey 
geo:facecol
    psychadelic with every triangle different colors for both instances and remainder 


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
        void writeVertexColors(GMergedMesh* mesh0, const GVolume* root);
   public:
        void setTarget(nvec3* target);  // where to write the colors
        void setRepeatIndex(unsigned int ridx);

   public:
        nvec3 make_color(unsigned int rgb);
   public:
        void traverse(const GVolume* root);   // full recursive traverse from root
   private:
        nvec3 getSurfaceColor(const GNode* node);

        void traverse_r( const GNode* node, unsigned depth );
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


