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

    void drawf(  int ix, int iy, int dx, int dy, float val); 
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



inline void SCanvas::drawf(int ix, int iy, int dx, int dy, float val)
{
    char tmp[10] ;
    int rc = sprintf(tmp, "%f", val );
    bool expect = rc == int(strlen(tmp)) ; 
    assert( expect );
    if(!expect) exit(EXIT_FAILURE) ; 

    _draw(ix, iy, dx, dy, tmp); 
}

inline void SCanvas::draw(int ix, int iy, int dx, int dy, int val)
{
    char tmp[10] ;
    int rc = sprintf(tmp, "%d", val );
    bool expect = rc == int(strlen(tmp)) ; 
    assert( expect );
    if(!expect) exit(EXIT_FAILURE) ; 

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
    if( ix < 0 ) ix += width ; 
    if( iy < 0 ) iy += height ;
    if( dx < 0 ) dx += xscale ; 
    if( dy < 0 ) dy += yscale ; 
 
    bool expect_ix =  ix >= 0 && ix < int(width)  ; 
    bool expect_iy =  iy >= 0 && iy < int(height) ; 
    bool expect_dx =  dx >= 0 && dx < int(xscale) ; 
    bool expect_dy =  dy >= 0 && dy < int(yscale) ; 

    bool expect = expect_ix && expect_iy && expect_dx && expect_dy ; 
    assert(expect); 
    if(!expect) exit(EXIT_FAILURE); 

    int x = ix*xscale + dx ; 
    int y = iy*yscale + dy ; 
    int l = strlen(txt) ; 

    bool expect_xy =  x + l < int(nx) &&  y < int(ny) ; 
    assert( expect_xy ); 

    if(!expect_xy) printf("SCanvas::_draw expect_xy ERROR out of range x+l %d  nx %d  y %d ny %d \n", x+l, nx, y, ny ); 
    if(!expect_xy) exit(EXIT_FAILURE); 


    int offset = y*nx + x ;  
    bool expect_offset = offset >= 0 && offset + l < int(nx*ny) ; 

    if(!expect_offset) printf("SCanvas::_draw error out of range offset+l %d  nx*ny %d \n", offset+l, nx*ny ) ; 
    assert(expect_offset);  
    if(!expect_offset) exit(EXIT_FAILURE); 
    

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

