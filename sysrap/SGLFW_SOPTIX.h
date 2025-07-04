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


/**
SGLFW_SOPTIX::render
-----------------------

1. mapping the interop buffer gives CUDA access into the OpenGL buffer via d_pixels
2. SOPTIX::render into d_pixels
3. unmap the interop buffer returning baton back to OpenGL
4. OpenGL display into the window

**/

inline void SGLFW_SOPTIX::render()
{
    uchar4* d_pixels = interop.output_buffer->map() ;
    ox.render(d_pixels);
    interop.output_buffer->unmap() ;
    interop.displayOutputBuffer(gl.window);
}





