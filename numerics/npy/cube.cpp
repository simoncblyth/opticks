
#define PXPYPZ {  1,  1,   1} 
#define PXPYMZ {  1,  1,  -1}
#define PXMYPZ {  1, -1,   1} 
#define PXMYMZ {  1, -1,  -1}
#define MXPYPZ { -1,  1,   1} 
#define MXPYMZ { -1,  1,  -1}
#define MXMYPZ { -1, -1,   1} 
#define MXMYMZ { -1, -1,  -1}


typedef struct {
    float x, y, z;
} point;

struct triangle 
{

  /*
   triangle(
       const point& a,  
       const point& b,  
       const point& c) 
   {
      pt[0] = a ; 
      pt[1] = b ; 
      pt[2] = c ; 
   } 
   */

   void copyTo(float* b)
   {
       memcpy(b+0, &pt[0], sizeof(float)*3);
       memcpy(b+3, &pt[1], sizeof(float)*3);
       memcpy(b+6, &pt[2], sizeof(float)*3);
   }
   point pt[3];
};


static quadrangle _cube[6] = {
     { {PXPYPZ, MXPYPZ, MXMYPZ, PXMYPZ }, },   
     { {PXPYPZ, PXMYPZ, PXMYMZ, PXPYMZ }, },   
     { {MXMYMZ, PXMYMZ, PXPYMZ, MXPYMZ }, },   
     { {MXMYMZ, MXPYMZ, MXPYPZ, MXMYPZ }, },   
     { {MXMYMZ, MXMYPZ, PXMYPZ, PXMYMZ }, }
};

float* cube_()
{
    float* buf = (float *)malloc(12*3*3*sizeof(float));
    for(int s = 0; s < 6; s++) 
    {
        quadrangle *q = &_cube[s];

        triangle t1 = q->tri1();
        triangle t2 = q->tri2();

        t1.copyTo(buf+s*3*3*2+0); 
        t2.copyTo(buf+s*3*3*2+3*3); 
    }
    return buf ;
}



