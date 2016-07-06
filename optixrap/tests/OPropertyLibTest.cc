#include "OPropertyLib.hh"

#include "OXPPNS.hh"

#include "OXRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    OXRAP_LOG__ ; 


    optix::Context context = optix::Context::create(); 

     




    return 0 ; 
}


/*

Attempt to isolate issue reported by Tao::

    124 2016-07-06 15:39:09.627 INFO  [17163] [Timer::operator@38] OpEngine:: START
    125 *** Error in `./OTracerTest': free(): invalid next size (fast): 0x000000000568bfe0 ***
    126 ======= Backtrace: =========
    127 /lib64/libc.so.6(+0x7275f)[0x7ff7d768675f]
    128 /lib64/libc.so.6(+0x77fce)[0x7ff7d768bfce]
    129 /lib64/libc.so.6(+0x78ce6)[0x7ff7d768cce6]
    130 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x61bbf3)[0x7ff7dc2b8bf3]
    131 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x675706)[0x7ff7dc312706]
    132 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x67aebd)[0x7ff7dc317ebd]
    133 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x67ae6a)[0x7ff7dc317e6a]
    134 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x676faa)[0x7ff7dc313faa]
    135 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x679818)[0x7ff7dc316818]
    136 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(+0x678a9d)[0x7ff7dc315a9d]
    137 /home/ihep/simon-dev-env/env-dev-2016july4/OptiX/lib64/liboptix.so.1(rtBufferUnmap+0x1eb)[0x7ff7dbcdb71b]
    138 libOptiXRap.so(+0x22efd)[0x7ff7d9fc7efd]
    139 libOptiXRap.so(_ZN12OPropertyLib6uploadERN5optix6HandleINS0_9BufferObjEEEP3NPYIfE+0x81)[0x7ff7d9fcc06d]
    140 libOptiXRap.so(_ZN16OScintillatorLib21makeReemissionTextureEP3NPYIfE+0x3d1)[0x7ff7d9fcdbff]
    141 libOptiXRap.so(_ZN16OScintillatorLib7convertEv+0xce)[0x7ff7d9fcd804]
    142 libOpticksOp.so(_ZN8OpEngine12prepareOptiXEv+0x5e8)[0x7ff7da440ec8]
    143 libGGeoView.so(_ZN3App12prepareOptiXEv+0xf7)[0x7ff7db0135ad]
    144 ./OTracerTest[0x403ab8]
    145 /lib64/libc.so.6(__libc_start_main+0xf5)[0x7ff7d7635b05]
    146 ./OTracerTest[0x403689]
    147 ======= Memory map: ========
    148 00400000-00408000 r-xp 00000000 08:08 859753560                          /home/ihep/simon-dev-e


*/
