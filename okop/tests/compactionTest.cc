#include "NGLM.hpp"
#include "NPY.hpp"
#include "OContext.hh"
#include "OBuf.hh"
#include "TBuf.hh"
#include "OXPPNS.hh"

#include "PLOG.hh"

#include "OXRAP_LOG.hh"
#include "NPY_LOG.hh"


/**

compactionTest
=================

Objective: download part of a GPU photon buffer (N,4,4) ie N*4*float4 
with minimal hostside memory allocation.

Thrust based approach:

* determine number of photons passing some criteria (eg with an associated PMT identifier)
* allocate temporary GPU hit_buffer and use thrust::copy_if to fill it  
* allocate host side hit_buffer sized appropriately and pull back the hits into it 

NB proof of concept code doing similar is in env-;optixthrust-

That was compilated by the 4*float4 perhaps can use a float4x4 type to
avoid complications ?

**/


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OXRAP_LOG__ ; 
    NPY_LOG__ ; 

    unsigned num_photons = 100 ; 
    unsigned PNUMQUAD = 4 ;
 
    NPY<float>* photon_data = NPY<float>::make(num_photons, PNUMQUAD, 4);
    photon_data->zero();   
    for(unsigned i=0 ; i < num_photons ; i++)
    {
        glm::uvec4 q0(i,i,i,i) ;
        glm::uvec4 q1(1000+i,1000+i,1000+i,1000+i) ;
        glm::uvec4 q2(2000+i,2000+i,2000+i,2000+i) ;
        glm::uvec4 q3(3000+i,3000+i,3000+i,3000+i) ;

        photon_data->setQuadU( q0, i, 0 );
        photon_data->setQuadU( q1, i, 1 );
        photon_data->setQuadU( q2, i, 2 );
        photon_data->setQuadU( q3, i, 3 );
    }


    optix::Context context = optix::Context::create();
    bool with_top = false ; 
    OContext ctx(context, OContext::COMPUTE, with_top);
    int entry = ctx.addEntry("compactionTest.cu.ptx", "compactionTest", "exception");

    optix::Buffer photon_buffer = context->createBuffer( RT_BUFFER_INPUT );
    photon_buffer->setFormat(RT_FORMAT_FLOAT4);
    //OBuf* pbuf = new OBuf("photon",photon_buffer);
    photon_buffer->setSize(num_photons*PNUMQUAD) ; 

    context[ "PNUMQUAD" ]->setUint( PNUMQUAD );   // quads per photon
    context["photon_buffer"]->setBuffer(photon_buffer);  
    ctx.launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  entry,  0, 0, NULL);

    OContext::upload<float>( photon_buffer, photon_data );

    ctx.launch( OContext::LAUNCH, entry, num_photons , 1, NULL ); 

/*
    CBufSpec sph = pbuf->bufspec();   // getDevicePointer happens here with OBufBase::bufspec

    TBuf tph("tph", sph );
*/




    return 0 ; 
}
