#pragma once

/**
SCanvas : ascii "painting"
==============================

Used for rendering CSG trees by ZSolid::Draw 

**/

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>

struct SCanvas
{
    bool verbose ; 
    unsigned width ; 
    unsigned height ; 
    unsigned xscale ; 
    unsigned yscale ; 
    unsigned nx ; 
    unsigned ny ; 
    char* c ; 

    SCanvas( unsigned width, unsigned height, unsigned xscale=8, unsigned yscale=4 );  
    void resize(unsigned width, unsigned height);
    void clear(); 
    void drawtest(); 

    void draw(   int ix, int iy, int dx, int dy, int val); 
    void drawch( int ix, int iy, int dx, int dy, char ch); 
    void draw(   int ix, int iy, int dx, int dy, const char* txt);
    void _draw(  int ix, int iy, int dx, int dy, const char* txt);      // 0,0 is at top left 

    void print(const char* msg=nullptr) const ; 
    const char* desc() const ; 
};


inline SCanvas::SCanvas(unsigned width_, unsigned height_, unsigned xscale_, unsigned yscale_)
    :
    verbose(getenv("VERBOSE")!=nullptr),
    xscale(xscale_), 
    yscale(yscale_),
    nx(0),
    ny(0),
    c(nullptr)
{
    resize(width_, height_); 
}

inline void SCanvas::resize(unsigned width_, unsigned height_)
{
    width = width_ ; 
    height = height_ ; 
    nx = width*xscale+1 ;   // +1 for the newline
    ny = height*yscale  ; 
    if(verbose)
    printf("SCanvas::resize width %d height %d nx %d ny %d nx*ny %d xscale %d yscale %d \n", width, height, nx, ny, nx*ny, xscale, yscale ); 
    delete [] c ; 
    c  = new char[nx*ny+1] ;   // +1 for string termination
    clear(); 
}

inline void SCanvas::clear()
{
    for(unsigned y=0 ; y < ny ; y++) for(unsigned x=0 ; x < nx ; x++)  c[y*nx+x] = ' ' ;   
    for(unsigned y=0 ; y < ny ; y++) c[y*nx+nx-1] = '\n' ;   
    c[nx*ny] = '\0' ;  // string terminate 
}

inline void SCanvas::drawtest()
{
    for(int ix=0 ; ix < int(width) ;  ix++ )
    for(int iy=0 ; iy < int(height) ; iy++ )
    {
        for(int dx=0 ; dx < int(xscale) ; dx++)
        for(int dy=0 ; dy < int(yscale) ; dy++)
        {
            draw(ix,iy,dx,dy, dx);
        }
    } 
}

inline void SCanvas::draw(int ix, int iy, int dx, int dy, int val)
{
    char tmp[10] ;
    int rc = sprintf(tmp, "%d", val );
    assert( rc == int(strlen(tmp)) );
    _draw(ix, iy, dx, dy, tmp); 
}

inline void SCanvas::drawch(int ix, int iy, int dx, int dy, char ch)  
{
    char tmp[2]; 
    tmp[0] = ch ; 
    tmp[1] = '\0' ; 
    _draw(ix, iy, dx, dy, tmp); 
}
inline void SCanvas::draw(int ix, int iy, int dx, int dy, const char* txt)   
{
    _draw(ix, iy, dx, dy, txt); 
}

inline void SCanvas::_draw(int ix, int iy, int dx, int dy, const char* txt)   // 0,0 is at top left 
{
    assert( ix < int(width)  ); 
    assert( iy < int(height)  ); 
    assert( dx < int(xscale) ); 
    assert( dy < int(yscale) ); 

    int x = ix*xscale + dx ; 
    int y = iy*yscale + dy ; 
    int l = strlen(txt) ; 

    if(!( x + l < int(nx) && y < int(ny) ))
    {
        printf("SCanvas::_draw error out of range x+l %d  nx %d  y %d ny %d \n", x+l, nx, y, ny ); 
        return ; 
    }

    int offset = y*nx + x ;  

    if(!(offset + l < int(nx*ny) ))
    {
        printf("SCanvas::_draw error out of range offset+l %d  nx*ny %d \n", offset+l, nx*ny ); 
        return ; 
    }

    memcpy( c + offset , txt, l );
}

inline void SCanvas::print(const char* msg) const 
{
    if(msg) printf("%s\n", msg); 
    if(verbose) 
        printf("\n[\n%s]\n",c);
    else 
        printf("\n%s",c);
}

inline const char* SCanvas::desc() const 
{
    char msg[200] ; 
    snprintf(msg, 200, "SCanvas::desc width %d height %d xscale %d yscale %d nx %d ny %d", width, height, xscale, yscale, nx, ny ); 
    return strdup(msg); 
}

