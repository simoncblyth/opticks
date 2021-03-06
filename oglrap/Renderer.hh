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

#include <cstddef>
#include <string>
#include <vector>
#include "plog/Severity.h"

#define QUERY_WORKAROUND 1

struct BBufSpec ; 
struct NSlice ; 

template <typename T> class NPY ; 

class Composition ;


struct RBuf ; 
struct RBuf4 ; 

class GDrawable ; 
class GMergedMesh ; 
class GBBoxMesh ; 
class GBuffer ;

class InstLODCull ; 
class Texture ; 

struct DrawElements ; 



#include "RendererBase.hh"
#include "OGLRAP_API_EXPORT.hh"


/**
Renderer
==========

Multiple flavors of Renderer are residents of Scene.

**/


class OGLRAP_API Renderer : public RendererBase  {
  public:

  static const plog::Severity LEVEL ;  
  static const char* PRINT ;  
  static const char* GMergedMesh_ ;
  static const char* GBBoxMesh_ ;
  static const char* Texture_ ;
  

  enum Attrib_IDs { 
                  vPosition=0, 
                     vColor=1, 
                    vNormal=2,
                  vTexcoord=3,
                 vTransform=4
    };

  enum { TEX_UNIT_0 , TEX_UNIT_1 } ;

  enum { MAX_LOD = 3 } ;

  public:
      Renderer(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      void setInstanced(bool instanced=true);
      void setInstLODCull(InstLODCull* instlodcull);
      bool isInstLODCullEnabled() const  ;
      void cull(); 

      void setWireframe(bool wireframe=true);
      virtual ~Renderer();
      std::string desc() const ; 
      std::string brief() const ; 
      void setLOD(int lod);
  private:
      void setNumLOD(int num_lod);
      void setType(const char* type);
  public: 
      //////////  CPU side buffer setup  ///////////////////
      /// HMM DOES THE RENDERER NEED TO KNOW THE DIFFERENCE BETWEEN THESE ?
      void upload(GBBoxMesh* bboxmesh);
      void upload(GMergedMesh* geometry);
      void upload(Texture* texture);
  private: 
      void setDrawable(GDrawable* drawable); 
      GBuffer* fslice_element_buffer(GBuffer* fbuf_orig, NSlice* fslice);
      bool hasTex(){ return m_has_tex ; }
      void setHasTex(bool hastex){ m_has_tex = hastex ; }
      bool hasTransforms(){ return m_has_transforms ; }
      void setHasTransforms(bool hastr){ m_has_transforms = hastr ; }
      void setupDraws(GMergedMesh* mm);
      void setupDrawsLOD(GMergedMesh* mm);
  private: 
      //////////  GPU side buffer setup  ///////////////////
      void upload();
      void setupInstanceFork();

  public: 
      void render();
      void setComposition(Composition* composition);
      Composition* getComposition(); 
  public: 
      void configureI(const char* name, std::vector<int> values);
      void dump(const char* msg="Renderer::dump");
      void Print(const char* msg="Renderer::Print");
      std::string locationString() const ;

  private:
      //void bind();
      void createVertexArrayLOD();
      GLuint createVertexArray(RBuf* instanceBuffer);

      void check_uniforms();
      void update_uniforms();   

      void dump(RBuf* buf);
      void dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count );

  private:
      GLuint m_vao[MAX_LOD] ; 
      GLuint m_vao_all  ; 

      DrawElements* m_draw[MAX_LOD] ; 

      bool          m_tagtex ; 
      unsigned      m_draw_0 ; 
      unsigned      m_draw_1 ; 
      unsigned      m_lod_counts[MAX_LOD] ; 
  private:

      RBuf*   m_vbuf ; 
      RBuf*   m_nbuf ; 
      RBuf*   m_cbuf ; 
      RBuf*   m_tbuf ; 
      RBuf*   m_fbuf ; 
      RBuf*   m_ibuf ; 

      RBuf4*  m_dst ; 
#ifdef QUERY_WORKAROUND
      RBuf4*  m_dst_devnull ; 
#endif
  private:
      // locations determined by *check_uniforms* and used by *update_uniforms* 
      // to update the "constants" available to shaders
      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_clip_location ;
      GLint  m_param_location ;
      GLint  m_scanparam_location ;
      GLint  m_nrmparam_location ;
      GLint  m_lightposition_location ;
#ifdef OLD_ITRANSFORM_LOC
      GLint  m_itransform_location ;
#endif
      GLint  m_colordomain_location ;
      GLint  m_colors_location ;
      GLint  m_pickface_location ;
      GLint  m_colorTex_location ;
      GLint  m_depthTex_location ;
   private:
      unsigned int m_itransform_count ;
      long         m_draw_count ;
      GLsizei      m_indices_count ;
  private:
      GDrawable*   m_drawable ;
      GMergedMesh* m_geometry ;
      GBBoxMesh*   m_bboxmesh ;
      Texture*     m_texture ;
      int          m_texture_id ; 
      Composition* m_composition ;
  private:
      bool m_has_tex ; 
      bool m_has_transforms ; 
      bool m_instanced ; 
      bool m_wireframe ; 

      InstLODCull* m_instlodcull ; 
      bool         m_instlodcull_enabled ; 
      int          m_num_lod ; 
      int          m_test_lod ; 
      bool         m_use_lod ; 
      int          m_lod ; 

      const char*  m_type ; 


};      



 
