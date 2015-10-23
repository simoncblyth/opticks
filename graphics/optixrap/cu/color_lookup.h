#pragma once

// see OBoundaryLib::convertColors GColors::setupCompositeColorBuffer

rtTextureSampler<uchar4,2>  color_texture  ;
rtDeclareVariable(uint4, color_domain, , );


static __device__ __inline__ uchar4 color_lookup(unsigned int line)
{
    return tex2D(color_texture, line+0.5f, 0.5f );  
}



