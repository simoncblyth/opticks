// name=CSGClassifyTest ; gcc $name.cc -std=c++11 -I.. -I$OPTICKS_PREFIX/include/sysrap -lstdc++ -o /tmp/$name && /tmp/$name 

#include <iostream>
#include "OpticksCSG.h"
#include "SCanvas.hh"

#define DEBUG 1 
#include "csg_classify.h"

/**

    +-------:+--------+
    |       :|        |
    |       :|        |
    +-------:+--------+


**/


void print_LUT( const LUT& lut, OpticksCSG_t typecode )
{
    unsigned width = 4+1 ; 
    unsigned height = 4+1 ; 
    unsigned xscale = 18 ; 
    unsigned yscale = 6 ; 

    int mx = 2 ; 
    int my = 2 ; 

    SCanvas canvas(width, height, xscale, yscale );

    canvas.draw(   0,0, mx, my, CSG::Name(typecode) ); 

    for(int ix=0 ; ix < width-1  ; ix++)
    for(int iy=0 ; iy < height-1 ; iy++)
    {
        canvas.drawch( ix,iy,    0,              0, '+' );
        canvas.drawch( ix,iy,    0,       yscale-1, '+' );

        for(int ixx=1 ; ixx < xscale ; ixx++ ) canvas.drawch( ix,iy,    ixx,   0, '-' );
        for(int iyy=1 ; iyy < yscale ; iyy++ ) canvas.drawch( ix,iy,      0, iyy, '|' );
    }    
   
    // rhs vertical 
    int ix = width-1 ; 
    for(int iy=0 ; iy < height-1 ; iy++)
    {
        canvas.drawch( ix,iy,    0,              0, '+' );
        for(int iyy=1 ; iyy < yscale ; iyy++ ) canvas.drawch( ix,iy,      0, iyy, '|' );
    }

    int iy = height-1 ; 
    for(int ix=0 ; ix < width-1 ; ix++)
    {
        canvas.drawch( ix,iy,    0,              0, '+' );
        for(int ixx=1 ; ixx < xscale ; ixx++ ) canvas.drawch( ix,iy,     ixx,  0, '-' );
    }
    canvas.drawch( width-1,height-1,   0,0, '+' );


    for(int a=0 ; a < 3 ; a++) 
    {
        IntersectionState_t A_state = (IntersectionState_t)a  ; 
        canvas.drawch( 0,a+1, mx+0,my+0, 'A' );
        canvas.draw(   0,a+1, mx+2,my+0, IntersectionState::Name(A_state) );
    }
    
    for(int b=0 ; b < 3 ; b++) 
    {
        IntersectionState_t B_state = (IntersectionState_t)b  ; 
        canvas.drawch( b+1,0,  mx+0,my+0, 'B' );
        canvas.draw(   b+1,0,  mx+2,my+0, IntersectionState::Name(B_state) );
    }

    for(int c=1 ; c < 3 ; c++)
    {
        bool A_closer = c == 1 ; 
        canvas.drawch( 0,0, mx+0, my+c,  A_closer ? 'A' : 'B' ); 
        canvas.draw(   0,0, mx+2, my+c, "Closer" ); 
    }

    for(int c=0 ; c < 2 ; c++)
    for(int b=0 ; b < 3 ; b++)
    for(int a=0 ; a < 3 ; a++)
    {
        bool A_closer = c == 0 ; 
        IntersectionState_t A_state = (IntersectionState_t)a  ; 
        IntersectionState_t B_state = (IntersectionState_t)b  ; 
        int ictrl = lut.lookup( typecode , A_state, B_state, A_closer ) ;
        const char* ctrl = CTRL::Name(ictrl) ; 
        assert( strlen(ctrl) < xscale ); 

        canvas.draw( b+1, a+1, mx+0, my+c+1,  ctrl ); 
    }
    canvas.print(); 
}


int main(int argc, char** argv)
{
    LUT lut ; 

    print_LUT(lut, CSG_UNION ); 
    print_LUT(lut, CSG_INTERSECTION ); 
    print_LUT(lut, CSG_DIFFERENCE); 

    return 0 ; 
}

/**

+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| union           | B Enter         | B Exit          | B Miss          |                 
| A Closer        |                 |                 |                 |                 
| B Closer        |                 |                 |                 |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Enter         |                 |                 |                 |                 
|                 | RETURN_A        | LOOP_A          | RETURN_A        |                 
|                 | RETURN_B        | RETURN_B        | RETURN_A        |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Exit          |                 |                 |                 |                 
|                 | RETURN_A        | RETURN_B        | RETURN_A        |                 
|                 | LOOP_B          | RETURN_A        | RETURN_A        |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Miss          |                 |                 |                 |                 
|                 | RETURN_B        | RETURN_B        | RETURN_MISS     |                 
|                 | RETURN_B        | RETURN_B        | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
      
     UNION FROM INSIDE                                   UNION FROM OUTSIDE                            
                                                                                                       
                                                                                                       
                  +-----------------------+                           +-----------------------+        
                  |                     B |                           |                     B |        
       +------------------+               |                +------------------+               |        
       | A        |       |               |                | A        |       |               |        
       |          |       |               |                |          |       |               |        
       |          |       |               |                |          |       |               |        
       |   0- - - 1 - - - 2 - - - - - - -[3]           0 - 1 - - - - [2]      |               |        
       |          |       |               |                |          |   3 - 4 - - - - - - -[5]       
       |          |       |               |                |          |       |               |        
       |          +-------|---------------+                |          +-------|---------------+        
       |                  |                                |                  |                        
       |                  |                                |                  |                        
       +------------------+                                +------------------+                        
                                                                                                       
                                                                                                       
     0: origin                                           0: origin                                     
     1: first B intersect, B Enter                       1: first A intersect, A Enter                 
     2: first A intersect, A Exit                        2: first B intersect, B Enter                 
     1,2: B Closer        ==> LOOP_B                     1,2: A Closer        ==> RETURN_A             
     3: second B intersect, B Exit                  
     2,3: A closer        ==> RETURN_B                   3: origin
                                                         4: first A intersect, A Exit
                                                         5: first B intersect, B Exit
                                                         4,5 A Closer         ==> RETURN_B

                                                                                    

+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| intersection    | B Enter         | B Exit          | B Miss          |                 
| A Closer        |                 |                 |                 |                 
| B Closer        |                 |                 |                 |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Enter         |                 |                 |                 |                 
|                 | LOOP_A          | RETURN_A        | RETURN_MISS     |                 
|                 | LOOP_B          | LOOP_B          | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Exit          |                 |                 |                 |                 
|                 | LOOP_A          | RETURN_A        | RETURN_MISS     |                 
|                 | RETURN_B        | RETURN_B        | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Miss          |                 |                 |                 |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
                                                                                          
                                                                                          
                                                                                          
                                                                                          
                                                                                          

+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| difference      | B Enter         | B Exit          | B Miss          |                 
| A Closer        |                 |                 |                 |                 
| B Closer        |                 |                 |                 |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Enter         |                 |                 |                 |                 
|                 | RETURN_A        | LOOP_A          | RETURN_A        |                 
|                 | LOOP_B          | LOOP_B          | RETURN_A        |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Exit          |                 |                 |                 |                 
|                 | RETURN_A        | LOOP_A          | RETURN_A        |                 
|                 | RETURN_FLIP_B   | RETURN_FLIP_B   | RETURN_A        |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Miss          |                 |                 |                 |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
                                                                                          

**/


