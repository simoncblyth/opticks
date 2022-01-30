// name=CSGClassifyTest ; gcc $name.cc -std=c++11 -I.. -I$OPTICKS_PREFIX/include/sysrap -lstdc++ -o /tmp/$name && /tmp/$name 

#include <iostream>
#include "OpticksCSG.h"
#include "SCanvas.hh"

#define DEBUG 1 
#include "csg_classify.h"

void print_LUT( const LUT& lut, OpticksCSG_t typecode, const char* notes )
{
    int width = 4+1 ; 
    int height = 4+1 ; 
    int xscale = 18 ; 
    int yscale = 6 ; 

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
        assert( strlen(ctrl) < unsigned(xscale) ); 

        canvas.draw( b+1, a+1, mx+0, my+c+1,  ctrl ); 
    }
    canvas.print(); 
    printf("%s\n", notes); 
}


const char* REFERENCE = R"LITERAL(

http://xrt.wikidot.com/doc:csg

http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf

A ray is shot at each sub-object to find the nearest intersection, then the
intersection with the sub-object is classified as one of entering, exiting or
missing it. Based upon the combination of the two classifications, one of
several actions is taken:

1. returning a hit
2. returning a miss
3. changing the starting point of the ray for one of the objects and then
   shooting this ray, classifying the intersection. In this case, the state
   machine enters a new loop.


Note that when there is a MISS on either side, the LUT results
are the same no matter what leftIsCloser is set to for all 
operations : UNION, INTERSECTION, DIFFERENCE.

)LITERAL" ;

const char* UNION = R"LITERAL(
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

(A Enter, B Enter) 
    origin outside both -> return closest

(A Exit, B Exit)   
    origin inside both  -> return furthest

(A Exit, B Enter)  
    origin inside A 

    * if A closer return A (means disjoint)     
    * if B closer loop B (to find the otherside of it, and then compare again)




     UX1 : UNION FROM INSIDE ONE                           UX2 : UNION FROM OUTSIDE 0,1,2                            
                                                           UX3 : UNION FROM INSIDE BOTH 3,4,5                                            
                                                                                                       
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
     1: B Enter                                          1: A Enter                 
     2: A Exit                                           2: B Enter                 
     1,2: B Closer        ==> LOOP_B                     1,2: A Closer        ==> RETURN_A             
     3: B Exit                  
     2,3: A closer        ==> RETURN_B                   3: origin
                                                         4: A Exit
                                                         5: B Exit
                                                         4,5 A Closer         ==> RETURN_B


      UX4 : DISJOINT UNION  0,[1],2

                                       
        +---------------------+                +---------------------+       
        | A                   |                |                   B |
        |                     |                |                     |
        |                     |                |                     |
        |                     |                |                     |
        |                     |                |                     |
        |         0- - - - - [1] -  - - - - - -2                     |
        |                     |                |                     |
        |                     |                |                     |
        |                     |                |                     |
        |                     |                |                     |
        +---------------------+                +---------------------+       


    0: origin
    1: A Exit
    2: B Enter
    1,2: A Closer        ==> RETURN_A [1]

)LITERAL" ;

                                                                        
const char* INTERSECTION = R"LITERAL(
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


     IX1 : INTERSECTION FROM INSIDE ONE 0,[1],2            IX2 : INTERSECTION FROM OUTSIDE 0,1,[2],3                            
                                                           IX3 : INTERSECTION FROM INSIDE BOTH 4,[5],6                                            
                                                                                                       
                  +-----------------------+                           +-----------------------+        
                  |                     B |                           |                     B |        
       +------------------+               |                +------------------+               |        
       | A        |       |               |                | A        |       |               |        
       |          |       |               |                |          |   4 -[5]- - - - - - - 6        
       |          |       |               |                |          |       |               |        
       |   0- - -[1]- - - 2               |            0 - 1 - - - - [2] - - -3               |        
       |          |       |               |                |          |       |               |        
       |          |       |               |                |          |       |               |        
       |          +-------|---------------+                |          +-------|---------------+        
       |                  |                                |                  |                        
       |                  |                                |                  |                        
       +------------------+                                +------------------+                        
                                                                                                       
                                                                                                       
     0: origin                                           0: origin                                     
     1: B Enter                                          1: A Enter                                            
     2: A Exit                                           2: B Enter                                           
     1,2: B Closer    ==> RETURN_B => [1]                1,2: A Closer   => LOOP_A  1->3 
                                                         3: A Exit
                                                         3,2: B Closer   => RETURN_B => [2]


                                                         4: origin
                                                         5: A Exit
                                                         6: B Exit
                                                         5,6: A Closer   ==> RETURN_A   [5]
)LITERAL" ;

const char* DIFFERENCE = R"LITERAL(
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



     DX1 : DIFFERENCE FROM INSIDE ONE 0,[1],2              DX2 : DIFFERENCE FROM OUTSIDE 0,[1],2                           
                                                           DX3 : DIFFERENCE FROM INSIDE BOTH 4,5,6                                            
                                                                                                       
                  +-----------------------+                           +-----------------------+        
                  |                     B |                           |                     B |        
       +------------------+               |                +------------------+               |        
       | A        |       |               |                | A        |       |               |        
       |          |       |               |                |          |   4 - 5 - - - - - - - 6        
       |          |       |               |                |          |       |               |        
       |   0- - -[1]> - - 2               |          0 - <[1] - - - - 2       |               |        
       |          |       |               |                |          |       |               |        
       |          |       |               |                |          |       |               |        
       |          +-------|---------------+                |          +-------|---------------+        
       |                  |                                |                  |                        
       |                  |                                |                  |                        
       +------------------+                                +------------------+                        
                                                                                                       
                                                                                                       
     0: origin                                           0: origin                                     
     1: B Enter                                          1: A Enter
     2: A Exit                                           2: B Enter
     1,2: B Closer  =>   RETURN_FLIP_B   [1]>            1,2: A Closer   => RETURN_A [1]


                                                         4: origin
                                                         5: A Exit
                                                         6: B Exit
                                                         5,6: A Closer    => LOOP_A  => A MISS   
                                                         ==> RETURN_MISS                                     

)LITERAL" ;

int main(int argc, char** argv)
{
    printf("%s\b", REFERENCE); 
    LUT lut ; 
    switch( argc > 1 ? argv[1][0] : 'A' )
    {
        case 'U': print_LUT(lut, CSG_UNION        , UNION );        break ;  
        case 'I': print_LUT(lut, CSG_INTERSECTION , INTERSECTION ); break ;  
        case 'D': print_LUT(lut, CSG_DIFFERENCE   , DIFFERENCE );   break ;  
        default: 
                  print_LUT(lut, CSG_UNION        , UNION );       
                  print_LUT(lut, CSG_INTERSECTION , INTERSECTION ); 
                  print_LUT(lut, CSG_DIFFERENCE   , DIFFERENCE );  
    }
    return 0 ; 
}

