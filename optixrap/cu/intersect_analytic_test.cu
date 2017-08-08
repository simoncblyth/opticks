#include "OpticksCSG.h"

// shape flag enums from npy-
#include "NPart.h"
#include "NCylinder.h"
#include "NSlab.h"
#include "NZSphere.h"

#include <optix_world.h>

#include "quad.h"
#include "Part.h"
#include "Prim.h"

#include "switches.h"
#define DEBUG 1

#include "boolean_solid.h"
#include "hemi-pmt.h"

// CUDART_ defines
#include "math_constants.h"

using namespace optix;

rtBuffer<Matrix4x4> tranBuffer; 

#include "bbox.h"

//#define CSG_INTERSECT_CONE_TEST 1
//#define CSG_INTERSECT_CONVEXPOLYHEDRON_TEST 1
#define CSG_INTERSECT_TORUS_TEST 1
#include "csg_intersect_primitive.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void intersect_analytic_test()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 

    rtPrintf("## intersect_analytic_test %llu\n", photon_id);

    //csg_intersect_cone_test(photon_id);
    //csg_intersect_convexpolyhedron_test(photon_id);
    csg_intersect_torus_test_0(photon_id);
    //csg_intersect_sphere_test(photon_id);
    
    output_buffer[photon_offset+0] = make_float4(40.f, 40.f, 40.f, 40.f);
    output_buffer[photon_offset+1] = make_float4(41.f, 41.f, 41.f, 41.f);
    output_buffer[photon_offset+2] = make_float4(42.f, 42.f, 42.f, 42.f);
    output_buffer[photon_offset+3] = make_float4(43.f, 43.f, 43.f, 43.f);

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();

    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 
    
    output_buffer[photon_offset+0] = make_float4(-40.f, -40.f, -40.f, -40.f);
    output_buffer[photon_offset+1] = make_float4(-41.f, -41.f, -41.f, -41.f);
    output_buffer[photon_offset+2] = make_float4(-42.f, -42.f, -42.f, -42.f);
    output_buffer[photon_offset+3] = make_float4(-43.f, -43.f, -43.f, -43.f);

}



