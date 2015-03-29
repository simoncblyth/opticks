#ifndef SCENE_H 
#define SCENE_H


class Shader ; 
class Camera ;
class View ;

// ggeo- coupling :
//    difficult to avoid the renderer knowing about the geometry
//    but attempting to keep the coupling weak 
//    by dealing in bytes and numbers of bytes
//
class GDrawable ; 
class GBuffer ;


class Scene {
  public:

  enum Attrib_IDs { vPosition=0, vColor=1 };

  public:
      Scene();
      virtual ~Scene();

  public: 
      void load(const char* envprefix);
      void init();
      void draw(int width, int height);

  public: 
      void dump(const char* msg="Scene::dump");
      void setShaderDir(const char* dir);
      char* getShaderDir(); 

  public: 
      void setGeometry(GDrawable* geometry);
      void setCamera(Camera* camera);
      void setView(View* view);

      float* getModelToWorld();
      GDrawable* getGeometry(); 
      Camera* getCamera(); 
      View* getView(); 

      void setupView(int width, int height);

  private:
      GLuint upload(GLenum target, GLenum usage, GBuffer* buffer);

  private:
      GLuint m_vao ; 
      GLuint m_program ;
      GLuint m_vertices ;
      GLuint m_colors ;
      GLuint m_indices ;
      GLint  m_mvp_location ;
      long   m_draw_count ;
      GLsizei m_indices_count ;
      float* m_model_to_world ; 

  private:
      GDrawable* m_geometry ;
      Camera* m_camera ;
      View*   m_view ;
      Shader* m_shader ;
      char* m_shaderdir ; 
};      


#endif



 
