#pragma once

// env/pycuda/pycuda_pyopengl_interop/pixel_buffer.py

#include "GMesh.hh"

struct Tex {
   Tex() : width(0), height(0), rgb(NULL), rgba(NULL) {}
   int width  ;
   int height ;
   unsigned char* rgb ;
   unsigned char* rgba ;
};


class Texture : public GMesh {
   public:
      static const float pvertex[] ;
      static const float pnormal[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;
      static const float ptexcoord[] ;

       Texture();
       void setSize(unsigned int width, unsigned int height);
       void loadPPM(char* path);  // does not need OpenGL context
       void create();
       //void resize(unsigned int width, unsigned int height, unsigned char* data);
       void cleanup();

       unsigned int getTextureId();
       unsigned int getSamplerId();
       unsigned int getWidth();
       unsigned int getHeight();

   private:
       void setup();
       void create_rgb(unsigned char* data);
       void create_rgba(unsigned char* data);

       unsigned int m_width ; 
       unsigned int m_height ; 
       unsigned int m_texture_id ; 
       unsigned int m_sampler_id ; 
       Tex          m_tex ; 

};


