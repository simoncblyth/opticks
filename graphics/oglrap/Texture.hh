#pragma once

// env/pycuda/pycuda_pyopengl_interop/pixel_buffer.py



#include "GMesh.hh"

class Texture : public GMesh {
   public:
      static const float pvertex[] ;
      static const float pnormal[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;
      static const float ptexcoord[] ;

       Texture();

       void create(unsigned int width, unsigned int height);
       void resize(unsigned int width, unsigned int height);
       void cleanup();
       void init();
       void draw();

       unsigned int getId();

   private:
       unsigned int m_width ; 
       unsigned int m_height ; 
       unsigned int m_id ; 

};


