#ifndef SCENE_H 
#define SCENE_H

// http://antongerdelan.net/opengl/hellotriangle.html
class Shader ; 

class Scene {
  public:
      static const float pvertex[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;

      Scene();
      virtual ~Scene();
      void init();
      void draw();

  private:
      GLuint m_vao ; 
      Shader* m_shader ;
      GLuint m_program ;
};      


#endif



 
