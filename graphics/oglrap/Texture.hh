#pragma once

// env/pycuda/pycuda_pyopengl_interop/pixel_buffer.py

#include "GMesh.hh"

struct Tex {
   Tex(bool zbuf_) : width(0), height(0), rgb(NULL), rgba(NULL), depth(NULL), zbuf(zbuf_) {}

   int width  ;
   int height ;
   unsigned char* rgb ;
   unsigned char* rgba ;
   unsigned char* depth ;
   bool zbuf ; 
};

// hmm Texture is too general, QuadTexture better
class Texture : public GMesh {
   public:
       Texture(bool zbuf=false);

       void loadPPM(const char* path);  // does not need OpenGL context
       void create();
       void cleanup();
   public:
       void setSize(unsigned int width, unsigned int height);
       unsigned int getId();
       unsigned int getWidth();
       unsigned int getHeight();
   private:
      static const float pvertex[] ;
      static const float pnormal[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;
      static const float ptexcoord[] ;
   private:
       unsigned int m_width ; 
       unsigned int m_height ; 
       unsigned int m_texture_id ; 
       Tex          m_tex ; 
};


inline void Texture::setSize(unsigned int width, unsigned int height)
{
    m_width = width ; 
    m_height = height ; 
}
inline unsigned int Texture::getId()
{
    return m_texture_id ;
}
inline unsigned int Texture::getWidth()
{
    return m_width ;
}
inline unsigned int Texture::getHeight()
{
    return m_height ;
}



