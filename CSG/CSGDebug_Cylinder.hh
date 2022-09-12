#pragma once
/**
CSGDebug_Cylinder.hh
======================

See notes/issues/ct_scan_nmskTailInner.rst

::

    c   # compile with DEBUG_CYLINDER macro 
    ./ct.sh 

    In [1]: d = s.CSGDebug_Cylinder     

    In [2]: d.shape
    Out[2]: (376, 8, 4)

    In [3]: d[0]
    Out[3]: 
    array([[  -264.   ,      0.   ,   -237.6  ,  69696.   ],
           [    -0.001,      0.   ,      1.   ,  56382.508],
           [  -264.   ,      0.   ,   -237.45 , 126078.51 ],
           [    -0.001,      0.   ,      1.   ,      1.   ],
           [     0.   ,      0.   ,      0.3  ,      0.09 ],
           [     0.3  ,    -71.235,   -237.277,     -0.323],
           [     0.   ,      0.016,     -0.   ,      0.   ],
           [    -0.   ,     -0.   ,     -1.   ,    237.277]], dtype=float32)

**/

#ifdef DEBUG_CYLINDER

#include "scuda.h"
#include "plog/Severity.h"
#include <vector>
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGDebug_Cylinder
{
    static constexpr const unsigned NUM_QUAD = 8u ; 
    static constexpr const char* NAME = "CSGDebug_Cylinder.npy" ; 

    static const plog::Severity LEVEL ; 
    static std::vector<CSGDebug_Cylinder> record ;     
    static void Save(const char* dir); 


    float3 ray_origin ;   // 0 
    float rr ; 

    float3 ray_direction ;  // 1 
    float k ; 

    float3 m ;             // 2 
    float mm ; 

    float3 n ;             // 3 
    float nn ;                    // d[:,3,3] all 1, when ray_direction normalized 

    float3 d ;             // 4 
    float dd ; 

    float nd ;             // 5        d[:,5,0]    sizeZ or -sizeZ (eg 0.3/-0.3 for the hz 0.15 thin one)    
    float md ; 
    float mn ;             //          d[:,5,2]
    float checkz ;         //          d[:,5,3]
 
    float a ;              // 6 
    float b ; 
    float c ;                           
    float disc ; 

    float4 isect ;         // 7
};

#endif
