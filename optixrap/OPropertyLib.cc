#include "NPY.hpp"
#include "OPropertyLib.hh"

#include "PLOG.hh"


OPropertyLib::OPropertyLib(optix::Context& ctx) : m_context(ctx)
{
}

void OPropertyLib::dumpVals( float* vals, unsigned int n)
{
    for(unsigned int i=0 ; i < n ; i++)
    { 
        std::cout << std::setw(10) << vals[i]  ;
        if(i % 16 == 0 ) std::cout << std::endl ; 
    }
}

void OPropertyLib::upload(optix::Buffer& optixBuffer, NPY<float>* buffer)
{
    unsigned int numBytes = buffer->getNumBytes(0) ;
    void* data = buffer->getBytes();
    memcpy( optixBuffer->map(), data, numBytes );
    optixBuffer->unmap(); 
}

void OPropertyLib::configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer)
{
    LOG(info) << "OPropertyLib::configureSampler" ; 

    // cuda-pdf p43 // default is to clamp to the range
    //RTwrapmode wrapmode = RT_WRAP_REPEAT ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ;
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ;  // return zero when out of range
    sampler->setWrapMode(0, wrapmode); 
    sampler->setWrapMode(1, wrapmode);


    //RTfiltermode filtermode = RT_FILTER_NEAREST ; 
    RTfiltermode filtermode = RT_FILTER_LINEAR ; 

    RTfiltermode minification = filtermode ; 
    RTfiltermode magnification = filtermode ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ;

    sampler->setFilteringModes(minification, magnification, mipmapping);

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ;
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ; 
    sampler->setReadMode(readmode);  

    RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // by inspection : zero based array index offset by 0.5
    //RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ; // needed by OptiX 400 ? see OptiX_400 pdf p17 
    sampler->setIndexingMode(indexingmode);  

   // with OptiX 400 cannot get any RT_TEXTURE_INDEX_ARRAY_INDEX to work ???

    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    // from pdf: OptiX currently supports only a single MIP level and a single element texture array.

    unsigned int texture_array_idx = 0u ;
    unsigned int mip_level = 0u ; 

    sampler->setBuffer(texture_array_idx, mip_level, buffer);
}


/*
(Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Unsupported combination of texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX, RT_WRAP_REPEAT, RT_FILTER_LINEAR, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp, line: 138)


*/



/*


  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetWrapMode sets the wrapping mode of
  * \a texturesampler to \a wrapmode for the texture dimension specified
  * by \a dimension.  \a wrapmode can take one of the following values:
  *
  *  - @ref RT_WRAP_REPEAT
  *  - @ref RT_WRAP_CLAMP_TO_EDGE
  *  - @ref RT_WRAP_MIRROR
  *  - @ref RT_WRAP_CLAMP_TO_BORDER
  *
  * The wrapping mode controls the behavior of the texture sampler as
  * texture coordinates wrap around the range specified by the indexing
  * mode.  These values mirror the CUDA behavior of textures.
  * See CUDA programming guide for details.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   dimension        Dimension of the texture
  * @param[in]   wrapmode         The new wrap mode of the texture sampler



 * @ref rtTextureSamplerSetFilteringModes sets the minification, magnification and MIP mapping filter modes for \a texturesampler.
  * RTfiltermode must be one of the following values:
  *
  *  - @ref RT_FILTER_NEAREST
  *  - @ref RT_FILTER_LINEAR
  *  - @ref RT_FILTER_NONE
  *
  * These filter modes specify how the texture sampler will interpolate
  * buffer data that has been attached to it.  \a minification and
  * \a magnification must be one of @ref RT_FILTER_NEAREST or
  * @ref RT_FILTER_LINEAR.  \a mipmapping may be any of the three values but
  * must be @ref RT_FILTER_NONE if the texture sampler contains only a
  * single MIP level or one of @ref RT_FILTER_NEAREST or @ref RT_FILTER_LINEAR
  * if the texture sampler contains more than one MIP level.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   minification     The new minification filter mode of the texture sampler
  * @param[in]   magnification    The new magnification filter mode of the texture sampler
  * @param[in]   mipmapping       The new MIP mapping filter mode of the texture sampler
  *




 * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetIndexingMode sets the indexing mode of \a texturesampler to \a indexmode.  \a indexmode
  * can take on one of the following values:
  *
  *  - @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
  *  - @ref RT_TEXTURE_INDEX_ARRAY_INDEX
  *
  * These values are used to control the interpretation of texture coordinates.  If the index mode is set to
  * @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES, the texture is parameterized over [0,1].  If the index
  * mode is set to @ref RT_TEXTURE_INDEX_ARRAY_INDEX then texture coordinates are interpreted as array indices
  * into the contents of the underlying buffer objects.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   indexmode        The new indexing mode of the texture sampler
  *




*/




/*

OptiX 400

2016-08-10 15:05:35.709 INFO  [1903430] [OContext::launch@214] OContext::launch
entry 0 width 1 height 1 libc++abi.dylib: terminating with uncaught exception
of type optix::Exception: Invalid value (Details: Function "RTresult
_rtContextValidate(RTcontext)" caught exception: Unsupported combination of
texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX,
RT_WRAP_REPEAT, RT_FILTER_LINEAR,
file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp,
line: 138) Abort trap: 6

*/






//
// NB this requires the memory layout of the optixBuffer needed for the texture of (nx,ny) shape
//    matches that of the NPY<float> buffer
//   
//    this was working for NPY<float> of shape    (128, 4, 39, 4)    
//         going into texture<float4>   nx:39  ny:128*4 boundaries*species
//
//    but for (128, 4, 39, 8)  it did not working 
//    as depends on the 39 being the last dimension of the buffer 
//    excluding the 4 that disappears as payload
//
//    maybe use
//             (128, 4, 39*2, 4 )
//    
//


