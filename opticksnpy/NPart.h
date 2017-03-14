#pragma once

enum { 
  PARAM_J  = 0, 
  PARAM_K  = 0 
};       // q0.f.xyzw

// (1,0) used for sizeZ in ZTubs // q1.u.x
enum { 
   INDEX_J    = 1, 
   INDEX_K    = 1 
};   // q1.u.y

enum { 
   BOUNDARY_J = 1, 
   BOUNDARY_K = 2 
};   // q1.u.z

enum { 
   FLAGS_J    = 1, 
   FLAGS_K    = 3 
};   // q1.u.w

enum { 
    BBMIN_J     = 2, 
    BBMIN_K     = 0 
};  // q2.f.xyz
enum { 
    TYPECODE_J  = 2, 
    TYPECODE_K  = 3 
};  // q2.u.w

enum { 
    BBMAX_J     = 3,     
    BBMAX_K = 0 
};  // q3.f.xyz 

enum { 
    NODEINDEX_J = 3, 
    NODEINDEX_K = 3 
};  // q3.u.w 


