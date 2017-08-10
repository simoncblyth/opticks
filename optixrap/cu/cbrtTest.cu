/*

This succeeds to reproduce the segv with intersect_analytic_test 
arising from SolveCubicNumericalRecipe.h cbrt(double).
Failing to reproduce the issue, despite attempts to converge 
header environment, compilation flags etc...

optixtest()
{
    # expects to be invoked from optixrap/cu 
    # and to find nam.cu ../tests/nam.cc

    local nam=${1:-cbrtTest}

    local exe=/tmp/$nam
    local ptx=/tmp/$nam.ptx

    local cc=../tests/$nam.cc
    local cu=$nam.cu

    local ver=OptiX_380
    #local ver=OptiX_400
    local inc=/Developer/$ver/include 
    local lib=/Developer/$ver/lib64 

    clang -std=c++11 -I/usr/local/cuda/include -I$inc -L$lib -loptix  -lc++  -Wl,-rpath,$lib  $cc  -o $exe

    #nvcc -arch=sm_30 -std=c++11 -O2 -use_fast_math -ptx $cu -I$inc -o $ptx
    #nvcc -arch=sm_30 -std=c++11  -use_fast_math -ptx $cu -I$inc -o $ptx
    nvcc -arch=sm_30  -use_fast_math -ptx $cu -I$inc -o $ptx
    #nvcc -arch=sm_30  -ptx $cu -I$inc -o $ptx

    echo $exe $ptx $nam
    $exe $ptx $nam

}
optixtest



::

    simon:cu blyth$ lldb /tmp/cbrtTest /tmp/cbrtTest.ptx cbrtTest
    (lldb) target create "/tmp/cbrtTest"
    Current executable set to '/tmp/cbrtTest' (x86_64).
    (lldb) settings set -- target.run-args  "/tmp/cbrtTest.ptx" "cbrtTest"
    (lldb) r
    Process 83172 launched: '/tmp/cbrtTest' (x86_64)
    Process 83172 stopped
    * thread #1: tid = 0x2c7ee4, 0x0000000100185124 liboptix.1.dylib`___lldb_unnamed_function1540$$liboptix.1.dylib + 36, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
        frame #0: 0x0000000100185124 liboptix.1.dylib`___lldb_unnamed_function1540$$liboptix.1.dylib + 36
    liboptix.1.dylib`___lldb_unnamed_function1540$$liboptix.1.dylib + 36:
    -> 0x100185124:  movq   %r15, (%rax)
       0x100185127:  jmp    0x10018512c               ; ___lldb_unnamed_function1540$$liboptix.1.dylib + 44
       0x100185129:  movq   %r15, (%rsi)
       0x10018512c:  movq   %rax, 0x8(%r15)
    (lldb) bt
    * thread #1: tid = 0x2c7ee4, 0x0000000100185124 liboptix.1.dylib`___lldb_unnamed_function1540$$liboptix.1.dylib + 36, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
      * frame #0: 0x0000000100185124 liboptix.1.dylib`___lldb_unnamed_function1540$$liboptix.1.dylib + 36
        frame #1: 0x0000000100185072 liboptix.1.dylib`___lldb_unnamed_function1539$$liboptix.1.dylib + 66
        frame #2: 0x0000000100369dc6 liboptix.1.dylib`___lldb_unnamed_function5838$$liboptix.1.dylib + 22
        frame #3: 0x0000000100248cc8 liboptix.1.dylib`___lldb_unnamed_function2623$$liboptix.1.dylib + 4920
        frame #4: 0x000000010024720e liboptix.1.dylib`___lldb_unnamed_function2618$$liboptix.1.dylib + 846
        frame #5: 0x0000000100246db7 liboptix.1.dylib`___lldb_unnamed_function2617$$liboptix.1.dylib + 279
        frame #6: 0x00000001002bd7ce liboptix.1.dylib`___lldb_unnamed_function3582$$liboptix.1.dylib + 590
        frame #7: 0x00000001002ddd22 liboptix.1.dylib`___lldb_unnamed_function4144$$liboptix.1.dylib + 194
        frame #8: 0x00000001002dd803 liboptix.1.dylib`___lldb_unnamed_function4143$$liboptix.1.dylib + 675
        frame #9: 0x00000001000d4884 liboptix.1.dylib`___lldb_unnamed_function972$$liboptix.1.dylib + 212
        frame #10: 0x000000010002e001 liboptix.1.dylib`rtProgramCreateFromPTXFile + 545
        frame #11: 0x00000001000034cc cbrtTest`optix::ContextObj::createProgramFromPTXFile(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 620
        frame #12: 0x00000001000015ed cbrtTest`OptiXMinimalTest::init(optix::Handle<optix::ContextObj>&) + 429
        frame #13: 0x0000000100001432 cbrtTest`OptiXMinimalTest::OptiXMinimalTest(optix::Handle<optix::ContextObj>&, char const*, char const*, char const*) + 98
        frame #14: 0x00000001000019e5 cbrtTest`OptiXMinimalTest::OptiXMinimalTest(optix::Handle<optix::ContextObj>&, char const*, char const*, char const*) + 53
        frame #15: 0x000000010000210a cbrtTest`main + 218
        frame #16: 0x00007fff8bf9a5fd libdyld.dylib`start + 1
    (lldb) f 10
    frame #10: 0x000000010002e001 liboptix.1.dylib`rtProgramCreateFromPTXFile + 545
    liboptix.1.dylib`rtProgramCreateFromPTXFile + 545:
    -> 0x10002e001:  movl   %eax, %r15d
       0x10002e004:  movl   %r15d, -0x84(%rbp)
       0x10002e00b:  callq  0x100100010               ; ___lldb_unnamed_function1233$$liboptix.1.dylib
       0x10002e010:  movq   %rax, %rbx
    (lldb) 



*/

#include <optix_world.h>

/*
#define SOLVE_QUARTIC_DEBUG 1
typedef double Solve_t ; 
#include "SolveCubicNumericalRecipe.h"
*/


#define SOLVE_QUARTIC_DEBUG 1
typedef double Solve_t ; 
#include "Solve.h"

/*
#include "quad.h"
#include "bbox.h"
#include "csg_intersect_torus.h"
*/


using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void cbrtTest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 
   

    float twentyseven_f = 27.f ; 
    float crf = cbrtf(twentyseven_f);

    double twentyseven_d(27) ; 
    double crd = cbrt(twentyseven_d);

    rtPrintf("cbrtTest crf:%f crd:%g  \n", crf, crd );



    Solve_t p,q,r ;   

    p = 49526.79994 ;        
    q = 408572956.1 ;
    r = -1483476.478 ;

    Solve_t roq = -r/q ; 
 
    rtPrintf("SolveCubicTest pqr (%15g %15g %15g)  x^3 + p x^2 + q x + r = 0   -r/q %g   \n", p,q,r, roq );
  

    Solve_t xx[3] ; 
    unsigned nr = SolveCubic(p,q,r,xx, 0u ); 
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

/*
    csg_intersect_torus_scale_test(photon_id, false);
    csg_intersect_torus_scale_test(photon_id, true );
*/


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

