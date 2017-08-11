#include <optix_world.h>
rtBuffer<rtCallableProgramId<unsigned(double,double,double,double*,unsigned)> > solve_callable ;

#include "quad.h"
#include "bbox.h"

#define CSG_INTERSECT_TORUS_TEST 1
#include "csg_intersect_torus.h"

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void intersect_analytic_test()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 

#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("// intersect_analytic_test %llu\n", photon_id);
#endif

    //csg_intersect_cone_test(photon_id);
    //csg_intersect_convexpolyhedron_test(photon_id);

    //  calling the below double laden function twice is prone to segv in createPTXFromFile
    //csg_intersect_torus_scale_test(photon_id, false);
    csg_intersect_torus_scale_test(photon_id, true );

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

