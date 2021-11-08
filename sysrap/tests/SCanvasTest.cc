// name=SCanvasTest ; gcc $name.cc -std=c++11 -I. -lstdc++ -o /tmp/$name && VERBOSE=1 /tmp/$name
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
    test_resize();  
    return 0 ; 
}

