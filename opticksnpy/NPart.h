#pragma once

enum { 
  PARAM_J  = 0, 
  PARAM_K  = 0 
};       // q0.f.xyzw


// only used for CSG operator nodes
enum {
    RTRANSFORM_J = 3,
    RTRANSFORM_K = 3 
};   // q3.u.w


enum {
    LEFT_J = 0, 
    LEFT_K = 3
};

enum {
    RIGHT_J = 1, 
    RIGHT_K = 3
};




// (1,0) used for sizeZ in ZTubs // q1.u.x
enum { 
   INDEX_J    = 1, 
   INDEX_K    = 1 
};   // q1.u.y

enum { 
   BOUNDARY_J = 1, 
   BOUNDARY_K = 2 
};   // q1.u.z


// only used for CSG operator nodes


enum { 
    BBMIN_J     = 2, 
    BBMIN_K     = 0 
};  // q2.f.xyz
enum { 
    TYPECODE_J  = 2, 
    TYPECODE_K  = 3 
};  // q2.u.w

enum { 
    BBMAX_J = 3,     
    BBMAX_K = 0 
};  // q3.f.xyz 

enum { 
    NODEINDEX_J = 3, 
    NODEINDEX_K = 3 
};  // q3.u.w 






// pointer arithmetic does not work on OptiX buffers
#define NPART_OFFSET(partFloatPtr, partOffset)  ( (partFloatPtr) + 16*(partOffset) )

#define NPART_Q0(partOffset)  ( 4*(partOffset) + 0 )
#define NPART_Q1(partOffset)  ( 4*(partOffset) + 1 )
#define NPART_Q2(partOffset)  ( 4*(partOffset) + 2 )
#define NPART_Q3(partOffset)  ( 4*(partOffset) + 3 )


#define NPART_TYPECODE(partFloatPtr) (  (partFloatPtr)+4*TYPECODE_J+TYPECODE_K )
#define NPART_NODEINDEX(partFloatPtr) ( (partFloatPtr)+4*NODEINDEX_J+NODEINDEX_K )




