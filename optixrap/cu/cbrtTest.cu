/*

For optixtest bash function to build and run see ../tests/OptixMinimalTest.hh 

This succeeds to reproduce the segv within createProgramFromPTXFile
within intersect_analytic_test arising from SolveCubicNumericalRecipe.h cbrt(double). 
But to do so had to almost duplicate it entirely.

* current working assumption is that double precision trig/cbrt etc functions
  are heavy on the GPU requiring a lot of code, hence there is a tendency for
  some particulary heavy functions like cbrt(double) to be the last straw
  that breaks the camels back

  * evidence for this came from cbrtTest.cu where repeating a function
    call triggered the segv

* a promising solution to this issue is to reuse heavy double precision  
  math functions like SolveCubic into RT_CALLABLE_PROGRAM to avoid the duplication

* https://devtalk.nvidia.com/default/topic/735307/optix/strange-error-while-reading-a-ptx-file/2
* https://devtalk.nvidia.com/default/topic/764148/double-precision-trigonometric-functions/#4282733
* https://devtalk.nvidia.com/search/more/sitecommentsearch/optix%20double%20precision/

*/

#include <optix_world.h>

rtBuffer<rtCallableProgramId<unsigned(double,double,double,double*,unsigned)> > callable ;


#define SOLVE_QUARTIC_DEBUG 1
typedef double Solve_t ; 
#include "SolveCubicNumericalRecipe.h"

/*
#define SOLVE_QUARTIC_DEBUG 1
typedef double Solve_t ; 
#include "Solve.h"
*/


/*
#include "quad.h"
#include "bbox.h"
#define CSG_INTERSECT_TORUS_TEST 1
#include "csg_intersect_torus.h"
*/


using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float4>  output_buffer;


RT_PROGRAM void cbrtTest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 
   

    unsigned msk = 0u ; 
    double a = 10.0 ; 
    double b = 20.0 ; 
    double c = 30.0 ; 

    double rts[3] ; 
    unsigned nrts ; 

    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 
    nrts = callable[0](a, b, c, rts, msk); 

#ifdef CSG_INTERSECT_TORUS_TEST 
    rtPrintf("cbrtTest SolveCubic_callable a:%f b:%f c:%f nrts:%u rts (%g %g %g)  \n", a,b,c,nrts,rts[0],rts[1],rts[2] );
#endif
   

    float twentyseven_f = 27.f ; 
    float crf = cbrtf(twentyseven_f);

    double twentyseven_d(27) ; 
    double crd = cbrt(twentyseven_d);

    Solve_t p,q,r ;   

    p = 49526.79994 ;        
    q = 408572956.1 ;
    r = -1483476.478 ;

    Solve_t roq = -r/q ; 

#ifdef CSG_INTERSECT_TORUS_TEST 
    rtPrintf("cbrtTest crf:%f crd:%g  \n", crf, crd );
    rtPrintf("SolveCubicTest pqr (%15g %15g %15g)  x^3 + p x^2 + q x + r = 0   -r/q %g   \n", p,q,r, roq );
#endif
 

    unsigned nr = 0 ;  
    Solve_t xx[3] ; 
    nr = SolveCubic(p,q,r,xx, 0u ); 
    nr = SolveCubic(p,q,r,xx, 0u ); 
    //nr = SolveCubic(p,q,r,xx, 0u ); 

    // NB : only one or 2 calls to simple inlined SolveCubic works 
    //      before getting segv in createProgramFromPTX 
    //      
    //      contrast with above callable approach where calling many times seems
    //      is not much of a resource burden
    //      
    //      It appears the heavy nature of GPU double precision math can be tamed 
    //      by doing it via callables. 
    //
    // HMM : doing twice works with default stacksize of 1024, more than twice segv in createProgramFromPTX
    //       three times works with 2* stacksize 


#ifdef CSG_INTERSECT_TORUS_TEST 
    rtPrintf("nr %u  \n", nr ) ;

    for(unsigned i=0 ; i < nr ; i++)
    {
        Solve_t x = xx[i] ; 

        Solve_t x3 = x*x*x ; 
        Solve_t x2 = p*x*x ; 
        Solve_t x1 = q*x ; 
        Solve_t x0 = r ; 

        Solve_t x3_x2 = x3 + x2 ; 
        Solve_t x1_x0 = x1 + x0 ;
        Solve_t x3_x2_x1_x0 = x3_x2 + x1_x0 ;
  
        Solve_t residual = ((x + p)*x + q)*x + r ; 
        rtPrintf("xx[%u] = %15g  residual %15g  x3210 (%15g %15g %15g %15g) x3_x2 %15g x1_x0 %15g x3_x2_x1_x0 %15g    \n", i, xx[i], residual, x3, x2, x1, x0, x3_x2, x1_x0, x3_x2_x1_x0 ) ;
    }
#endif


    // WOW: calling "csg_intersect_torus_scale_test" twice is prone to: Segmentation fault: 11   within createPTXFromFile
    //csg_intersect_torus_scale_test(photon_id, false);
    //csg_intersect_torus_scale_test(photon_id, true);


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

