#pragma once

#include <vector>

class GDrawable ; 
class GBuffer ;
class Composition ;

#include "RendererBase.hh"

class Renderer : public RendererBase  {
  public:

  static const char* PRINT ;  
  enum Attrib_IDs { 
        vPosition=0, 
           vColor=1, 
          vNormal=2,
          vTexcoord=3
    };

  public:
      Renderer(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      virtual ~Renderer();

  public: 
      void setDrawable(GDrawable* drawable, bool debug=false);
      void render();
      void setComposition(Composition* composition);
      Composition* getComposition(); 

  public: 
      void configureI(const char* name, std::vector<int> values);
      void dump(const char* msg="Renderer::dump");
      void Print(const char* msg="Renderer::Print");

  private:
      void gl_upload_buffers(bool debug);
      GLuint upload(GLenum target, GLenum usage, GBuffer* buffer);

      bool hasTex(){ return m_has_tex ; }
      void setHasTex(bool hastex){ m_has_tex = hastex ; }

      void check_uniforms();
      void update_uniforms();   

      void dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count );


  private:
      GLuint m_vao ; 

      GLuint m_vertices ;
      GLuint m_normals ;
      GLuint m_colors ;
      GLuint m_texcoords ;
      GLuint m_indices ;

      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_clip_location ;
      GLint  m_param_location ;

      long   m_draw_count ;
      GLsizei m_indices_count ;

  private:
      GDrawable* m_drawable ;
      Composition* m_composition ;
      bool m_has_tex ; 
};      


inline Renderer::Renderer(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_texcoords(0),
    m_mv_location(-1),
    m_mvp_location(-1),
    m_clip_location(-1),
    m_param_location(-1),
    m_draw_count(0),
    m_indices_count(0),
    m_drawable(NULL),
    m_composition(NULL),
    m_has_tex(false)
{
}





inline void Renderer::setComposition(Composition* composition)
{
    m_composition = composition ;
}
inline Composition* Renderer::getComposition()
{
    return m_composition ;
}



 
