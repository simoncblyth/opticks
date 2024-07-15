#pragma once
/**
SIMG_Frame.h
=============

Connector between SIMG.h image handling and 
for example SGLFW.h OpenGL umbrella. 

**/

#define SIMG_IMPLEMENTATION 1
#include "SIMG.h"
#include "ssys.h"

struct SIMG_Frame
{
    int width ; 
    int height ;
    int channels ; 
    int num_pixels ; 
    size_t size ;     
    int quality ; 

    unsigned char* pixels ; 
    SIMG*      img ; 

    SIMG_Frame(int _width, int _height);

    void flipVertical(); 
    void writeJPG(const char* dir, const char* name) const ; 
    void writeJPG(const char* path) const ; 
};

inline SIMG_Frame::SIMG_Frame(int _width, int _height)
    :
    width(_width),
    height(_height),
    channels(4),
    num_pixels(width*height),
    size(width*height*channels),
    quality(ssys::getenvint("SIMG_Frame__QUALITY", 50)),
    pixels(new unsigned char[size]),
    img(new SIMG(width, height, channels, pixels ))
{
}

inline void SIMG_Frame::flipVertical()
{
    img->flipVertical(); 
}
inline void SIMG_Frame::writeJPG(const char* dir, const char* name) const 
{
    img->writeJPG(dir, name, quality );  
}
inline void SIMG_Frame::writeJPG(const char* path) const 
{
    img->writeJPG(path, quality );  
}



