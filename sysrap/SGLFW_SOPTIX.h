#pragma once

#include "SOPTIX.h"
#include "SGLFW_CUDA.h"

struct SGLFW_SOPTIX
{
    SGLFW& gl ;
    SGLM& gm ;

    SOPTIX ox ;
    SGLFW_CUDA interop ; // interop buffer display coordination

    SGLFW_SOPTIX( SGLFW& gl );
    void render();
};

inline SGLFW_SOPTIX::SGLFW_SOPTIX( SGLFW& _gl )
    :
    gl(_gl),
    gm(gl.gm),
    ox(gm),
    interop(gm)
{
}

inline void SGLFW_SOPTIX::render()
{
    uchar4* d_pixels = interop.output_buffer->map() ; // map buffer : give access to CUDA
    ox.render(d_pixels);
    interop.output_buffer->unmap() ;                  // unmap buffer : end CUDA access, back to OpenGL
    interop.displayOutputBuffer(gl.window);
}





