#include "Demo.hh"
#include "stdio.h"
#include "assert.h"

#include "GVector.hh"

/*
 
           y ^
             |
             R
             | 
             |
      -------|--------> x
             |
             |
        B    |    G 
 

*/


#define XOFF  100.0f
#define YOFF  100.0f
#define ZOFF  100.0f
#define VAL    10.0f

const float Demo::pvertex[] = {
   0.0f+XOFF,  VAL+YOFF,  0.0f+ZOFF,
    VAL+XOFF, -VAL+YOFF,  0.0f+ZOFF,
   -VAL+XOFF, -VAL+YOFF,  0.0f+ZOFF
};

const float Demo::pcolor[] = {
  1.0f, 0.0f,  0.0f,
  0.0f, 1.0f,  0.0f,
  0.0f, 0.0f,  1.0f
};

const float Demo::pnormal[] = {
  0.0f, 0.0f,  1.0f,
  0.0f, 0.0f,  1.0f,
  0.0f, 0.0f,  1.0f
};

const float Demo::ptexcoord[] = {
  0.0f, 0.0f,
  1.0f, 0.0f,
  1.0f, 1.0f
};




const unsigned int Demo::pindex[] = {
      0,  1,  2
};


Demo::Demo() : GMesh(0, (gfloat3*)&pvertex[0],3, (guint3*)&pindex[0],1, (gfloat3*)&pnormal[0], (gfloat2*)&ptexcoord[0]) 
{
    setColors( (gfloat3*)&pcolor[0] );
}

Demo::~Demo()
{
}


