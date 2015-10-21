#pragma once

#include <vector>

class GDrawable ; 
class GMergedMesh ; 
class GBBoxMesh ; 
class Texture ; 

class GBuffer ;
class Composition ;

struct NSlice ; 

#include "RendererBase.hh"

class Renderer : public RendererBase  {
  public:

  static const char* PRINT ;  
  enum Attrib_IDs { 
        vPosition=0, 
           vColor=1, 
          vNormal=2,
          vTexcoord=3,
          vTransform=4
    };

  public:
      Renderer(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      void setInstanced(bool instanced=true);
      void setWireframe(bool wireframe=true);
      virtual ~Renderer();

  public: 
      void upload(GMergedMesh* geometry, bool debug=false);
      void upload(GBBoxMesh* bboxmesh, bool debug=false);
      void upload(Texture* texture, bool debug=false);
  public: 
      void bind();
      void render();
      void setComposition(Composition* composition);
      Composition* getComposition(); 
  public: 
      void configureI(const char* name, std::vector<int> values);
      void dump(const char* msg="Renderer::dump");
      void Print(const char* msg="Renderer::Print");

  private:
      void upload_buffers(NSlice* slice);
      GLuint upload(GLenum target, GLenum usage, GBuffer* buffer, const char* name=NULL);

      bool hasTex(){ return m_has_tex ; }
      void setHasTex(bool hastex){ m_has_tex = hastex ; }
      bool hasTransforms(){ return m_has_transforms ; }
      void setHasTransforms(bool hastr){ m_has_transforms = hastr ; }

      void check_uniforms();
      void update_uniforms();   

      void dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count );

  private:
      GLuint m_vao ; 
  private:
      // buffer id
      GLuint m_vertices ;
      GLuint m_normals ;
      GLuint m_colors ;
      GLuint m_texcoords ;
      GLuint m_indices ;
      GLuint m_transforms ;
  private:
      // locations determined by *check_uniforms* and used by *update_uniforms* 
      // to update the "constants" available to shaders
      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_clip_location ;
      GLint  m_param_location ;
      GLint  m_nrmparam_location ;
      GLint  m_lightposition_location ;
      GLint  m_itransform_location ;
      GLint  m_colordomain_location ;
      GLint  m_colors_location ;
      GLint  m_pickface_location ;
   private:
      unsigned int m_itransform_count ;
      long         m_draw_count ;
      GLsizei      m_indices_count ;
  private:
      GDrawable*   m_drawable ;
      GMergedMesh* m_geometry ;
      GBBoxMesh*   m_bboxmesh ;
      Texture*     m_texture ;
      Composition* m_composition ;
  private:
      bool m_has_tex ; 
      bool m_has_transforms ; 
      bool m_instanced ; 
      bool m_wireframe ; 
};      


inline Renderer::Renderer(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_texcoords(0),
    m_mv_location(-1),
    m_mvp_location(-1),
    m_clip_location(-1),
    m_param_location(-1),
    m_nrmparam_location(-1),
    m_lightposition_location(-1),
    m_itransform_location(-1),
    m_colordomain_location(-1),
    m_colors_location(-1),
    m_pickface_location(-1),
    m_itransform_count(0),
    m_draw_count(0),
    m_indices_count(0),
    m_drawable(NULL),
    m_geometry(NULL),
    m_bboxmesh(NULL),
    m_texture(NULL),
    m_composition(NULL),
    m_has_tex(false),
    m_has_transforms(false),
    m_instanced(false),
    m_wireframe(false)
{
}


inline void Renderer::setInstanced(bool instanced)
{
    m_instanced = instanced ; 
}
inline void Renderer::setWireframe(bool wireframe)
{
    m_wireframe = wireframe ; 
}
inline void Renderer::setComposition(Composition* composition)
{
    m_composition = composition ;
}
inline Composition* Renderer::getComposition()
{
    return m_composition ;
}



 
