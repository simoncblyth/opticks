#include <GL/glew.h>

#include "Colors.hh"
#include "Device.hh"

#include "NPY.hpp"
#include "PLOG.hh"
// trace/debug/info/warning/error/fatal


Colors::Colors(Device* device)
    :
    m_device(device),
    m_colors_tex(0),
    m_colors_uploaded(false),
    m_colorbuffer(NULL)
{
}

void Colors::setColorBuffer(NPY<unsigned char>* colorbuffer)
{
    m_colorbuffer = colorbuffer ; 
}


unsigned int Colors::getNumColors()
{
    return m_colorbuffer ? m_colorbuffer->getNumItems() : 1 ; 
}

void Colors::upload()
{
    if(!m_colorbuffer)
    {
         LOG(warning) <<"Colors::upload no colorbuffer skipping " ;  
         return ; 
    }

    if(m_colors_uploaded)
    {
         LOG(warning) <<"Colors::upload already uploaded " ;  
         return ; 
    }
    m_colors_uploaded = true ; 

    unsigned int ncol = getNumColors() ;
    if(ncol == 0)
    {
         LOG(warning) <<"Colors::upload empty colorbuffer " ;  
         return ; 
    }


    // moving from GBuffer to NPY<unsigned char> changes shape from (ncol, 1) -> (ncol, 4)
    // ie the big bitshift uchar4 combination lives in the new buffer as separate uchar 
    //

    //unsigned char* colors = (unsigned char*)m_colorbuffer->getPointer();
    unsigned char* colors = m_colorbuffer->getValues();
    LOG(debug) <<"Colors::upload ncol " << ncol ;  

    // https://open.gl/textures
    GLenum  target = GL_TEXTURE_1D ;   // Must be GL_TEXTURE_1D or GL_PROXY_TEXTURE_1D

    glGenTextures(1, &m_colors_tex);
    glBindTexture(target, m_colors_tex);

    GLint   level = 0 ;                // level-of-detail number, Level 0 is the base image level
    GLint   internalFormat = GL_RGBA  ;  // number of color components in the texture
    GLsizei width = ncol ;                // width of the texture image including the border if any (powers of 2 are better)
    GLint   border = 0 ;               // width of the border. Must be either 0 or 1
    GLenum  format = GL_RGBA ;         // format of the pixel data
    GLenum  type = GL_UNSIGNED_BYTE ;  // type of the pixel data

    glTexImage1D(target, level, internalFormat, width, border, format, type, colors );

    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

}

/*
With ncol = 5 and using WRAP_S: GL_CLAMP_TO_EDGE, MIN_FILTER : GL_NEAREST, MAG_FILTER : GL_NEAREST
With colors  R G M Y C  scanned float values to probe the texture algo find...

      
        -0.10  R
     ---------------
         0.00  R 
         0.200 R
     ---------------
         0.201 G 
         0.205 G 
         ...
         0.400 G  
     ---------------
         0.401 M
         0.5   M
         0.6   M 
     ---------------
         0.601 Y
         0.7   Y
         0.8   Y
     --------------
         0.801 C
         0.9   C
         1.0   C
     --------------
         1.1   C
         1.5   C    

So an n+1 binning approach is used...

In [2]: np.linspace(0.,1.,5+1 )
Out[2]: array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
                RRRRRRRRGGGGGGMMMMMM
 
So to pick via an integer would need  (float(i) + 0.5)/5.0  to land mid-bin, where i ranges 0 -> (5-1)

In [10]: (np.arange(0,5) + 0.5)/5.           
Out[10]: array([ 0.1,  0.3,  0.5,  0.7,  0.9])

For 1-based indices i 1->5 need

        (float(i) - 1.0 + 0.5)/5.0 

*/




