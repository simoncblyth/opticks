// name=SCanvasTest ; gcc $name.cc -std=c++11 -I.. -lstdc++ -Wsign-compare -o /tmp/$name && VERBOSE=1 /tmp/$name
#include "SCanvas.hh"

void test_draw_int()
{
    int xscale = 5 ; 
    int yscale = 3 ; 
    int width = 10 ; 
    int height = 5 ; 

    SCanvas* c = new SCanvas(width,height,xscale,yscale); 

    for(int ix=0 ; ix < width ;  ix++ )
    for(int iy=0 ; iy < height ; iy++ )
    {
        for(int dx=0 ; dx < xscale ; dx++)
        for(int dy=0 ; dy < yscale ; dy++)
        {
            c->draw(ix,iy,dx,dy, dx);
        }
    } 
    c->print(); 
}

void test_draw_txt()
{
    int xscale = 5 ; 
    int yscale = 3 ; 
    int width = 10 ; 
    int height = 5 ; 

    SCanvas* c = new SCanvas(width,height,xscale,yscale); 

    for(int ix=0 ; ix < width ;  ix++ )
    for(int iy=0 ; iy < height ; iy++ )
    {
        int dx = 0 ; 
        for(int dy=0 ; dy < yscale ; dy++)
        {
            c->draw(ix,iy,dx,dy, "txt01");  
        }
    } 
    c->print(); 
}

void test_format_float()
{
    float offset = 5e6 ; 

    char tmp[16] ;
    for( float val=0.f ; val < 10.f ; val+= 0.50001f )
    {
        int len = sprintf(tmp, "%7.2f", val + offset );
        bool expect = len == int(strlen(tmp)) ; 
        printf("//test_format_float  tmp:%s  len:%d  expect:%d offset %g  \n", tmp, len, expect, offset ); 
    }
}

void test_draw_float()
{
    int xscale = 8 ; 
    int yscale = 3 ; 
    int width = 10 ; 
    int height = 5 ; 

    SCanvas* c = new SCanvas(width,height,xscale,yscale); 

    for(int ix=0 ; ix < width ;  ix++ )
    for(int iy=0 ; iy < height ; iy++ )
    {
        int dx = 0 ; 
        for(int dy=0 ; dy < yscale ; dy++)
        {
            float val = float(dy) + 0.5f ; 
            c->drawf(ix,iy,dx,dy, val );  
        }
    } 
    c->print(); 
}

void test_resize()
{
    int xscale = 5 ; 
    int yscale = 3 ; 
    int width = 10 ; 
    int height = 5 ; 

    SCanvas* c = new SCanvas(width,height,xscale,yscale); 
    c->drawtest(); 
    c->print(); 

    c->resize(width*2, height*2) ; 
    c->drawtest(); 
    c->print(); 

    c->resize(width, height) ; 
    c->drawtest(); 
    c->print(); 
}

int main(int argc, char** argv)
{
    //test_draw_int(); 
    //test_draw_txt(); 
    //test_resize();  

    test_format_float(); 
    //test_draw_float(); 
 
    return 0 ; 
}

