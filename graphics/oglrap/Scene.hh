#ifndef SCENE_H 
#define SCENE_H

// http://antongerdelan.net/opengl/hellotriangle.html
class Shader ; 
class IGeometry ; 
class Buffer ;

class Scene {

  enum Attrib_IDs { vPosition=0, vColor=1 };

  public:
      Scene();
      virtual ~Scene();
      void init(IGeometry* geometry);
      void draw();
      void dump(const char* msg="Scene::dump");
      void setShaderDir(const char* dir);
      char* getShaderDir(); 

  private:
      GLuint upload(GLenum target, GLenum usage, Buffer* buffer);

  private:
      GLuint m_vao ; 
      GLuint m_program ;
      GLuint m_vertices ;
      GLuint m_colors ;
      GLuint m_indices ;
      GLint  m_nelem ;

  private:
      Shader* m_shader ;
      char* m_shaderdir ; 
};      


#endif



 
