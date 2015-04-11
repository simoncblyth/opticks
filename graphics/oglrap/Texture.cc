#include "Texture.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "assert.h"

#define LOADPPM_IMPLEMENTATION
#include "loadPPM.h"



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
    m_texture_id(0),
    m_sampler_id(0),
    m_tex()
{
    setColors( (gfloat3*)&pcolor[0] );
}

void Texture::loadPPM(char* path)
{
    m_tex.data = ::loadPPM(path, &m_tex.width, &m_tex.height );
    if(m_tex.data)
    {
        printf("Texture::loadPPM loaded %s into tex of width %d height %d \n", path, m_tex.width, m_tex.height);
        setSize(m_tex.width, m_tex.height);
    }
    else
    {
        printf("Texture::loadPPM failed to load %s \n", path );
    }
}

void Texture::create()
{
    assert(m_tex.data);
    create(m_tex.data);
}

void Texture::setSize(unsigned int width, unsigned int height)
{
    m_width = width ; 
    m_height = height ; 
}


void Texture::create(unsigned char* data)
{
    assert(m_width > 0 && m_height > 0);
    glGenTextures(1, &m_texture_id);
    glBindTexture( GL_TEXTURE_2D, m_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data );


   // glGenSamplers(1, &m_sampler_id);
   // GLuint sampler_unit = 0 ; 
   // glBindSampler(sampler_unit, sampler_id); 

}

void Texture::cleanup()
{
    glDeleteTextures(1, &m_texture_id);
    m_texture_id = 0 ;
}


/*
void Texture::resize(unsigned int width, unsigned int height, unsigned char* data)
{
    if(width == m_width && height == m_height) return ;
    cleanup();
    create(data);
}
*/


unsigned int Texture::getTextureId()
{
    return m_texture_id ;
}
unsigned int Texture::getSamplerId()
{
    return m_sampler_id ;
}



unsigned int Texture::getWidth()
{
    return m_width ;
}
unsigned int Texture::getHeight()
{
    return m_height ;
}



