#ifndef RENDERER_H 
#define RENDERER_H

#include <vector>

class Shader ; 
class Composition ;
class GDrawable ; 
class GBuffer ;


class Renderer {
  public:

  static const char* PRINT ;  
  enum Attrib_IDs { 
        vPosition=0, 
           vColor=1, 
          vNormal=2,
          vTexcoord=3
    };

  public:
      Renderer(const char* tag);
      virtual ~Renderer();

  public: 
      void setDrawable(GDrawable* drawable);
      void setComposition(Composition* composition);
      void setShaderDir(const char* dir);
      void setShaderTag(const char* tag);

  public: 
      void render();

  public: 
      void configureI(const char* name, std::vector<int> values);
      void dump(const char* msg="Renderer::dump");
      void Print(const char* msg="Renderer::Print");

      Composition* getComposition(); 
      char* getShaderDir(); 
      char* getShaderTag(); 

  private:
      void gl_upload_buffers();
      GLuint upload(GLenum target, GLenum usage, GBuffer* buffer);

      bool hasTex(){ return m_has_tex ; }
      void setHasTex(bool hastex){ m_has_tex = hastex ; }

  private:
      GLuint m_vao ; 
      GLuint m_program ;

      GLuint m_vertices ;
      GLuint m_normals ;
      GLuint m_colors ;
      GLuint m_texcoords ;
      GLuint m_indices ;

      GLint  m_mv_location ;
      GLint  m_mvp_location ;
      GLint  m_sampler_location ;

      long   m_draw_count ;
      GLsizei m_indices_count ;

  private:
      GDrawable* m_drawable ;
      Composition* m_composition ;
      Shader* m_shader ;
      char* m_shaderdir ; 
      char* m_shadertag ; 
      bool m_has_tex ; 
};      


#endif

 
