#pragma once


/*

          quad q1 ; 
            q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];      // (nodeIdx-1) as 1-based
            OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float4& v, int i)
{
  return ((float*)(&v))[i];
}

#define PART_TYPECODE( partBufferFloatPtr, partOffsetUInt, nodeIdx1UInt ) ( (partBuffer)[4*((partOffset)+(nodeIdx)-1) + TYPECODE_J] 

static __device__
float unsigned_as_float(unsigned u)
{
  union {
    float f;
    unsigned u;
  } v1;

  v1.u = u;
  return v1.f;
}



*/



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



// pointer arithmetic does not work on OptiX buffers
#define NPART_OFFSET(partFloatPtr, partOffset)  ( (partFloatPtr) + 16*(partOffset) )

#define NPART_Q0(partOffset)  ( 4*(partOffset) + 0 )
#define NPART_Q1(partOffset)  ( 4*(partOffset) + 1 )
#define NPART_Q2(partOffset)  ( 4*(partOffset) + 2 )
#define NPART_Q3(partOffset)  ( 4*(partOffset) + 3 )


#define NPART_TYPECODE(partFloatPtr) (  (partFloatPtr)+4*TYPECODE_J+TYPECODE_K )
#define NPART_NODEINDEX(partFloatPtr) ( (partFloatPtr)+4*NODEINDEX_J+NODEINDEX_K )




