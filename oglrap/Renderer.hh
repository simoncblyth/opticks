#pragma once

#include <cstddef>
#include <vector>

struct BBufSpec ; 
struct NSlice ; 

template <typename T> class NPY ; 

class Composition ;

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

  static const char* PRINT ;  
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
      void setWireframe(bool wireframe=true);
      virtual ~Renderer();

  public: 
      //////////  CPU side buffer setup  ///////////////////
      void upload(GBBoxMesh* bboxmesh, bool debug=false);
      void upload(GMergedMesh* geometry, bool debug=false);
      void upload(Texture* texture, bool debug=false);
  private: 
      void setTexture(Texture* texture);
      Texture* getTexture() const ; 
       void setDrawable(GDrawable* drawable); 
      GBuffer* fslice_element_buffer(GBuffer* fbuf_orig, NSlice* fslice);
      bool hasTex(){ return m_has_tex ; }
      void setHasTex(bool hastex){ m_has_tex = hastex ; }
      bool hasTransforms(){ return m_has_transforms ; }
      void setHasTransforms(bool hastr){ m_has_transforms = hastr ; }
      void setupDraws(GMergedMesh* mm);
  private: 
      //////////  GPU side buffer setup  ///////////////////
      void upload_GBuffer(GLenum target, GLenum usage, GBuffer* buffer, const char* name=NULL);
      void upload_NPY(    GLenum target, GLenum usage, NPY<float>* buffer, const char* name=NULL);
      GLuint upload(        GLenum target, GLenum usage, BBufSpec* bufspec, const char* name=NULL);
      void upload();

  public: 
      void bind();
      void render();
      void setComposition(Composition* composition);
      Composition* getComposition(); 
  public: 
  public: 
      void configureI(const char* name, std::vector<int> values);
      void dump(const char* msg="Renderer::dump");
      void Print(const char* msg="Renderer::Print");

  private:

      GLuint createVertexArray(NPY<float>* ibuf);

      void check_uniforms();
      void update_uniforms();   

      void dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count );

  private:
      GLuint m_vao ; 
      DrawElements* m_draw[MAX_LOD] ; 
      unsigned      m_draw_num ; 
      unsigned      m_draw_0 ; 
      unsigned      m_draw_1 ; 
  private:
      GBuffer*  m_vbuf ;  // verts
      GBuffer*  m_nbuf ;  // normals
      GBuffer*  m_cbuf ;  // colors
      GBuffer*  m_tbuf ;  // textures
      GBuffer*  m_fbuf ;  // faces/element/indices 
      NPY<float>*  m_ibuf ;  // instance transforms 
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
      GLint  m_itransform_location ;
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


};      



 
