#pragma once
/**
scanvas.h : ascii "painting" (formerly SCanvas.hh without .cc)
==================================================================

Used for rendering CSG trees by ZSolid::Draw 

::

    epsilon:opticks blyth$ opticks-f scanvas.h 
    ./CSG/tests/CSGClassifyTest.cc:#include "scanvas.h"
    ./CSG/CSGDraw.cc:#include "scanvas.h"
    ./extg4/X4SolidTree.cc:#include "scanvas.h"
    ./sysrap/CMakeLists.txt:    scanvas.h
    ./sysrap/tests/scanvasTest.cc:#include "scanvas.h"
    ./sysrap/tests/TreePruneTest.cc:#include "scanvas.h"
    ./sysrap/scanvas.h:scanvas.h : ascii "painting" (formerly SCanvas.hh without .cc)
    ./u4/U4SolidTree.cc:#include "scanvas.h"
    epsilon:opticks blyth$ 

**/

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>

struct scanvas
{
    bool verbose ; 
    unsigned width ; 
    unsigned height ; 
    unsigned xscale ; 
    unsigned yscale ; 
    unsigned nx ; 
    unsigned ny ; 
    char* c ; 

    scanvas( unsigned width, unsigned height, unsigned xscale=8, unsigned yscale=4 );  
    void resize(unsigned width, unsigned height);
    void clear(); 
    void drawtest(); 

    void drawf(  int ix, int iy, int dx, int dy, float val , const char* fmt="%7.2f" ); 
    void draw(   int ix, int iy, int dx, int dy, int val); 
    void drawch( int ix, int iy, int dx, int dy, char ch); 
    void draw(   int ix, int iy, int dx, int dy, const char* txt);
    void _draw(  int ix, int iy, int dx, int dy, const char* txt);      // 0,0 is at top left 

    void print(const char* msg=nullptr) const ; 
    const char* desc() const ; 
};


inline scanvas::scanvas(unsigned width_, unsigned height_, unsigned xscale_, unsigned yscale_)
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

inline void scanvas::resize(unsigned width_, unsigned height_)
{
    width = width_ ; 
    height = height_ ; 
    nx = width*xscale+1 ;   // +1 for the newline
    ny = height*yscale  ; 
    if(verbose)
    printf("scanvas::resize width %d height %d nx %d ny %d nx*ny %d xscale %d yscale %d \n", width, height, nx, ny, nx*ny, xscale, yscale ); 
    delete [] c ; 
    c  = new char[nx*ny+1] ;   // +1 for string termination
    clear(); 
}

inline void scanvas::clear()
{
    for(unsigned y=0 ; y < ny ; y++) for(unsigned x=0 ; x < nx ; x++)  c[y*nx+x] = ' ' ;   
    for(unsigned y=0 ; y < ny ; y++) c[y*nx+nx-1] = '\n' ;   
    c[nx*ny] = '\0' ;  // string terminate 
}

inline void scanvas::drawtest()
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



inline void scanvas::drawf(int ix, int iy, int dx, int dy, float val, const char* fmt )
{
    char tmp[16] ;
    int rc = sprintf(tmp, fmt, val );
    bool expect = rc == int(strlen(tmp)) ; 
    assert( expect );
    if(!expect) exit(EXIT_FAILURE) ; 

    bool expect_xscale = xscale > 7 ; 
    if(!expect_xscale) printf("scanvas::_draw expect_xscale when drawing floats an xscale of at least 8 is needed  xscale %d  \n", xscale) ; 


    _draw(ix, iy, dx, dy, tmp); 
}

inline void scanvas::draw(int ix, int iy, int dx, int dy, int val)
{
    char tmp[16] ;
    int rc = sprintf(tmp, "%d", val );
    bool expect = rc == int(strlen(tmp)) ; 
    assert( expect );
    if(!expect) exit(EXIT_FAILURE) ; 

    _draw(ix, iy, dx, dy, tmp); 
}

inline void scanvas::drawch(int ix, int iy, int dx, int dy, char ch)  
{
    char tmp[2]; 
    tmp[0] = ch ; 
    tmp[1] = '\0' ; 
    _draw(ix, iy, dx, dy, tmp); 
}
inline void scanvas::draw(int ix, int iy, int dx, int dy, const char* txt)   
{
    _draw(ix, iy, dx, dy, txt); 
}

inline void scanvas::_draw(int ix, int iy, int dx, int dy, const char* txt)   // 0,0 is at top left 
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

    if(!expect) printf("scanvas::_draw ix %d width %d iy %d height %d dx %d xscale %d dy %d yscale %d \n", ix, width, iy, height, dx, xscale, dy, yscale ); 
    if(!expect) return ;  

    //assert(expect); 
    //if(!expect) exit(EXIT_FAILURE); 
    //if(!expect) exit(EXIT_FAILURE); 

    int x = ix*xscale + dx ; 
    int y = iy*yscale + dy ; 
    int l = strlen(txt) ; 

    bool expect_x =  x + l < int(nx) ; 
    bool expect_y =  y < int(ny) ; 

    if(!expect_x) printf("scanvas::_draw expect_x ERROR out of range ix %d xscale %d dx %d x %d l %d x+l %d  nx %d txt [%s] \n", ix, xscale, dx, x, l, x+l, nx, txt ); 
    if(!expect_y) printf("scanvas::_draw expect_y ERROR out of range y %d ny %d \n",   y, ny ); 

    //if(!expect_x) exit(EXIT_FAILURE); 
    //if(!expect_y) exit(EXIT_FAILURE); 

    if(!expect_x) return ; 
    if(!expect_y) return ; 



    int offset = y*nx + x ;  
    bool expect_offset = offset >= 0 && offset + l < int(nx*ny) ; 

    if(!expect_offset) printf("scanvas::_draw error out of range offset+l %d  nx*ny %d \n", offset+l, nx*ny ) ; 

    //assert(expect_offset);  
    //if(!expect_offset) exit(EXIT_FAILURE); 
    if(!expect_offset) return ;     


    memcpy( c + offset , txt, l );
}

inline void scanvas::print(const char* msg) const 
{
    if(msg) printf("%s\n", msg); 
    if(verbose) 
        printf("\n[\n%s]\n",c);
    else 
        printf("\n%s",c);
}

inline const char* scanvas::desc() const 
{
    char msg[200] ; 
    snprintf(msg, 200, "scanvas::desc width %d height %d xscale %d yscale %d nx %d ny %d", width, height, xscale, yscale, nx, ny ); 
    return strdup(msg); 
}

