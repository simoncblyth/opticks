#include "Texture.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
/*
   "1" (-1,1) [0,1]          "0" (1,1) [1,1]



   "2" (-1,-1) [0,0]         "3" (1,-1) [1,0]
*/


const float Texture::pvertex[] = {  
     1.0f,  1.0f, 0.0f, 
    -1.0f,  1.0f, 0.0f, 
    -1.0f, -1.0f, 0.0f, 
     1.0f, -1.0f, 0.0f
};

const float Texture::pcolor[] = { 
      1.0f, 0.0f, 0.0f,  
      0.0f, 1.0f, 0.0f,  
      0.0f, 0.0f, 1.0f,  
      1.0f, 0.0f, 0.0f
};  
                              
const float Texture::pnormal[] = { 
      0.0f, 0.0f, 1.0f,  
      0.0f, 0.0f, 1.0f,  
      0.0f, 0.0f, 1.0f,  
      0.0f, 0.0f, 1.0f
};  
 
const float Texture::ptexcoord[] = { 
     1.0f, 1.0f, 
     0.0f, 1.0f, 
     0.0f, 0.0f, 
     1.0f, 0.0f 
};


const unsigned int Texture::pindex[] = {
      0,  1,  2 ,
      2,  3,  0 
};




Texture::Texture() : 
    GMesh(0, 
             (gfloat3*)&pvertex[0],
             4, 
             (guint3*)&pindex[0],
             2, 
             (gfloat3*)&pnormal[0],
             (gfloat2*)&ptexcoord[0]
         ),
    m_id(0)
{
    setColors( (gfloat3*)&pcolor[0] );
}





void Texture::create(unsigned int width, unsigned int height)
{
    m_width = width ; 
    m_height = height ; 

    glGenTextures(1, &m_id);
    glBindTexture( GL_TEXTURE_2D, m_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
}

void Texture::cleanup()
{
    glDeleteTextures(1, &m_id);
    m_id = 0 ;
}


void Texture::resize(unsigned int width, unsigned int height)
{
    if(width == m_width && height == m_height) return ;
    cleanup();
    create(width, height);
}


unsigned int Texture::getId()
{
    return m_id ;
}


