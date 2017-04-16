#pragma once

/*

Currently:

    q0 : param       <-- primitive use
    q1 : xy:prim        <-- primitive use
    ------------------------------------------------
    q2 : xyz:bbox 
    q3 : xyz:bbox     

Aiming for:

    q0 : param    <-- primitive use
    q1 : param1   <-- primitive use
    ------------------------------------------------
    q2 :  
    q3 :     


Hmm supporting partlist with its bbox makes it 
difficult to rejig the layout, so for now just shift the id to w

2017-04-16 16:44:17.307 INFO  [513893] [GParts::dump@857] GParts::dump ni 4
     0.0000      0.0000      0.0000   1000.0000 
     0.0000       0 <-id       123 <-bnd       0.0000  bn Rock//perfectAbsorbSurface/Vacuum 
     0.0000      0.0000      0.0000           6 (box) TYPECODE 
     0.0000      0.0000      0.0000           0 (nodeIndex) 

     0.0000      0.0000      0.0000      0.0000 
     0.0000       1 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000           2 (intersection) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 

     0.0000      0.0000      0.0000    500.0000 
     0.0000       2 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000           5 (sphere) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 

     0.0000      0.0000      1.0000      0.0000 
  -100.0000       3 <-id       124 <-bnd       0.0000  bn Vacuum///GlassSchottF2 
     0.0000      0.0000      0.0000          13 (slab) TYPECODE 
     0.0000      0.0000      0.0000           1 (nodeIndex) 



*/




enum { 
  PARAM_J  = 0, 
  PARAM_K  = 0 
};       // q0.f.xyzw

enum { 
  PARAM1_J  = 1, 
  PARAM1_K  = 0 
};       // q1.f.xyzw



/*
enum {
    LEFT_J = 0, 
    LEFT_K = 3
};
enum {
    RIGHT_J = 1, 
    RIGHT_K = 3
};
*/





// only used for CSG operator nodes in input serialization buffer
enum {
    TRANSFORM_J = 3,
    TRANSFORM_K = 3 
};   // q3.u.w

// only used for CSG primitives in part buffer
enum {
    GTRANSFORM_J = 3,
    GTRANSFORM_K = 0 
};   // q3.u.x





//enum { INDEX_J    = 1, INDEX_K    = 1  };   // q1.u.y
enum { INDEX_J    = 1, INDEX_K    = 3  };   // q1.u.w
enum { BOUNDARY_J = 1, BOUNDARY_K = 2  };   // q1.u.z







enum { 
    NODEINDEX_J = 3, 
    NODEINDEX_K = 3 
};  // q3.u.w 

enum { 
    TYPECODE_J  = 2, 
    TYPECODE_K  = 3 
};  // q2.u.w







// only used for CSG operator nodes
enum { 
    BBMIN_J     = 2, 
    BBMIN_K     = 0 
};  // q2.f.xyz
enum { 
    BBMAX_J = 3,     
    BBMAX_K = 0 
};  // q3.f.xyz 




/*
// pointer arithmetic does not work on OptiX buffers
#define NPART_OFFSET(partFloatPtr, partOffset)  ( (partFloatPtr) + 16*(partOffset) )

#define NPART_Q0(partOffset)  ( 4*(partOffset) + 0 )
#define NPART_Q1(partOffset)  ( 4*(partOffset) + 1 )
#define NPART_Q2(partOffset)  ( 4*(partOffset) + 2 )
#define NPART_Q3(partOffset)  ( 4*(partOffset) + 3 )


#define NPART_TYPECODE(partFloatPtr) (  (partFloatPtr)+4*TYPECODE_J+TYPECODE_K )
#define NPART_NODEINDEX(partFloatPtr) ( (partFloatPtr)+4*NODEINDEX_J+NODEINDEX_K )

*/


