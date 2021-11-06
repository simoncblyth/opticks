#pragma once
#include <cassert>
#include <cstring>
#include <cstdio>

struct SCanvas
{
    unsigned xscale ; 
    unsigned yscale ; 
    unsigned nx ; 
    unsigned ny ; 
    char* c ; 

    SCanvas( unsigned width, unsigned height, unsigned xscale=8, unsigned yscale=4 );  
    void clear(); 
    void draw(int ix, int iy, int dy, int val); 
    void drawch(int ix, int iy, int dy, char ch); 
    void draw(int ix, int iy, int dy, const char* txt);
    void _draw(int x, int y, int dy, const char* txt);      // 0,0 is at top left 

    void print(const char* msg=nullptr) const ; 
};


SCanvas::SCanvas(unsigned width, unsigned height, unsigned xscale_, unsigned yscale_)
    :
    xscale(xscale_), 
    yscale(yscale_),  
    nx(width*(xscale+1)),
    ny(height*(yscale+1)),
    c(new char[nx*ny+1])  // +1 for string termination
{
    clear(); 
}

inline void SCanvas::clear()
{
    for(unsigned y=0 ; y < ny ; y++) for(unsigned x=0 ; x < nx ; x++)  c[y*nx+x] = ' ' ;   
    for(unsigned y=0 ; y < ny ; y++) c[y*nx+nx-1] = '\n' ;   
    c[nx*ny] = '\0' ;  // string terminate 
}


inline void SCanvas::draw(int ix, int iy, int dy, int val)
{
    char tmp[10] ;
    int rc = sprintf(tmp, "%d", val );
    assert( rc == int(strlen(tmp)) );
    _draw(ix, iy, dy, tmp); 
}

inline void SCanvas::drawch(int ix, int iy, int dy, char ch)  
{
    char tmp[2]; 
    tmp[0] = ch ; 
    tmp[1] = '\0' ; 
    _draw(ix, iy, dy, tmp); 
}
inline void SCanvas::draw(int ix, int iy, int dy, const char* txt)   
{
    _draw(ix, iy, dy, txt); 
}

inline void SCanvas::_draw(int ix, int iy, int dy, const char* txt)   // 0,0 is at top left 
{
    int x = ix*xscale ; 
    int y = iy*yscale + dy ; 

    memcpy( c + y*nx + x , txt, strlen(txt) );
    // memcpy to avoid string termination 
    // hmm: drawing near the righthand side may stomp on newlines
    // hmm: drawing near bottom right risks writing off the end of the canvas
}

inline void SCanvas::print(const char* msg) const 
{
    if(msg) printf("%s\n", msg); 
    printf("\n%s",c);
}



